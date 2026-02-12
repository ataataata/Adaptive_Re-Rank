import math
import time
import ir_datasets
import pandas as pd
import torch
import nltk
import os
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker
from tqdm import tqdm 

lambda_cost = 0.05 #you can adjust the lambda here

# Setup
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
STEMMER = nltk.SnowballStemmer("english")

#creating inverted index
class InvertedIndex:
    def __init__(self):
        self.postings = defaultdict(list)
        self.doc_lens = {}
        self.num_docs = 0
        self.avg_dl = 0

    def add_document(self, docid, tokens):
        if not tokens: return
        self.num_docs += 1
        self.doc_lens[docid] = len(tokens)
        counts = Counter(tokens)
        for term, tf in counts.items():
            self.postings[term].append((docid, tf))

    def finalize(self):
        self.avg_dl = sum(self.doc_lens.values()) / self.num_docs if self.num_docs > 0 else 0

def preprocess(text):
    if not text: return []
    tokens = nltk.word_tokenize(text.lower())
    return [STEMMER.stem(t) for t in tokens if t.isalnum() and t not in STOPWORDS]

def score_bm25(query_tokens, index, k1=1.5, b=0.75):
    doc_scores = defaultdict(float)
    for term in query_tokens:
        if term not in index.postings: continue
        postings = index.postings[term]
        df = len(postings)
        idf = math.log((index.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        for docid, tf in postings:
            dl = index.doc_lens[docid]
            denom = tf + k1 * (1 - b + b * (dl / index.avg_dl))
            doc_scores[docid] += idf * (tf * (k1 + 1) / denom)
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

def calculate_metrics(run, qrels, k=10):
    dcg, mrr, found_mrr = 0.0, 0.0, False
    ideal_rels = sorted(qrels.values(), reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
    
    for i, (doc_id, _) in enumerate(run[:k]):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)
            if not found_mrr:
                mrr = 1.0 / (i + 1)
                found_mrr = True
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"ndcg": ndcg, "mrr": mrr, "eff": 0.5 * ndcg + 0.5 * mrr}

def run_labeling_pipeline(dataset_id, name, lambda_cost):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce_l6 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device) #if you would like to use another light re-reanker change here
    bge_reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=(device == "cuda")) #if you would like to use another heavy re-reanker change here
    
    ds = ir_datasets.load(dataset_id)
    index = InvertedIndex()
    corpus = {}
    
    print(f"\n--- Indexing {name} ---")
    docs = list(ds.docs_iter())
    for d in tqdm(docs, desc="Building Inverted Index"):
        txt = f"{getattr(d, 'title', '')} {getattr(d, 'text', '')}".strip()
        corpus[d.doc_id] = txt
        index.add_document(d.doc_id, preprocess(txt))
    index.finalize()

    qrels = defaultdict(dict)
    for qr in ds.qrels_iter():
        qrels[qr.query_id][qr.doc_id] = qr.relevance

    results = []
    # Filter queries to only those that have qrels
    query_list = [q for q in ds.queries_iter() if q.query_id in qrels]

    print(f"\n--- Ranking & Labeling {name} ---")
    for q_obj in tqdm(query_list, desc="Processing Queries"):
        qid, q_text = q_obj.query_id, q_obj.text
        q_tokens = preprocess(q_text)
        if not q_tokens: continue 

        # BM25 Stage
        t0 = time.perf_counter()
        bm25_run = score_bm25(q_tokens, index)[:50]
        lat_bm25 = time.perf_counter() - t0
        
        # L6 Stage (Pipeline = Retrieval + Rerank)
        t_start_l6 = time.perf_counter()
        l6_pairs = [(q_text, corpus[did]) for did, _ in bm25_run]
        l6_scores = ce_l6.predict(l6_pairs) if l6_pairs else []
        l6_run = sorted(zip([d[0] for d in bm25_run], l6_scores), key=lambda x: x[1], reverse=True)
        lat_l6 = (time.perf_counter() - t_start_l6) + lat_bm25
        
        # BGE Stage (Pipeline = Retrieval + Rerank)
        t_start_bge = time.perf_counter()
        bge_pairs = [[q_text, corpus[did]] for did, _ in bm25_run]
        bge_scores = bge_reranker.compute_score(bge_pairs) if bge_pairs else []
        bge_run = sorted(zip([d[0] for d in bm25_run], bge_scores), key=lambda x: x[1], reverse=True)
        lat_bge = (time.perf_counter() - t_start_bge) + lat_bm25
        
        # Metrics
        m_bm25 = calculate_metrics(bm25_run, qrels[qid])
        m_l6 = calculate_metrics(l6_run, qrels[qid])
        m_bge = calculate_metrics(bge_run, qrels[qid])
        
        # Utilities (Used for labeling, but raw data is saved for re-calc)
        max_lat = max(lat_bm25, lat_l6, lat_bge)
        def get_u(eff, lat):
            return (1.0 - lambda_cost) * eff + lambda_cost * (1.0 - (lat / max_lat if max_lat > 0 else 0))

        u_bm25, u_l6, u_bge = get_u(m_bm25['eff'], lat_bm25), get_u(m_l6['eff'], lat_l6), get_u(m_bge['eff'], lat_bge)
        
        options = [("bm25", u_bm25, 0), ("l6", u_l6, 1), ("bge", u_bge, 2)]
        best_name, _, best_cls = max(options, key=lambda x: x[1])

        results.append({
            "dataset": name, "qid": qid, "query": q_text, "class": best_cls, "best_model": best_name,
            "utility_bm25": u_bm25, "utility_l6": u_l6, "utility_bge": u_bge,
            "eff_bm25": m_bm25['eff'], "eff_l6": m_l6['eff'], "eff_bge": m_bge['eff'],
            "lat_bm25": lat_bm25, "lat_l6": lat_l6, "lat_bge": lat_bge
        })

    # Save Data
    os.makedirs("labeled_queries", exist_ok=True)
    df = pd.DataFrame(results)
    csv_filename = os.path.join("labeled_queries", f"labels_{name}.csv")
    df.to_csv(csv_filename, index=False)

    # Append to Stats (create file if it doesn't exist)
    class_counts = Counter(df['class'])
    if not os.path.exists("stats.txt"):
        with open("stats.txt", "w") as f:
            f.write("Labeling Run Log\n==========================\n\n")
    with open("stats.txt", "a") as f:
        f.write(f"--- Dataset: {name} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        f.write(f"Total Queries: {len(df)}\n")
        f.write(f"Class 0 (BM25): {class_counts.get(0, 0)}\n")
        f.write(f"Class 1 (L6):   {class_counts.get(1, 0)}\n")
        f.write(f"Class 2 (BGE):  {class_counts.get(2, 0)}\n")
        f.write(f"CSV Path: {csv_filename}\n\n")

if __name__ == "__main__":
    datasets_to_run = [
        #("beir/fiqa/train", "fiqa-train")
        #("beir/msmarco/train", "msmarco-train")
        #("beir/nfcorpus/train", "nfcorpus-train")
        #("beir/scifact/train", "scifact-train")
        #("beir/trec-covid", "trec-covid")

        """
            you can pick whichever datasets exist in ir_datasets libary for your own labelling
        """

    ]

    for ds_id, ds_name in datasets_to_run:
        run_labeling_pipeline(ds_id, ds_name)
import os
import torch
import numpy
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertTokenizer,
    BertModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# ============================================================
# 1. Load labeled data
#    Reads all CSV files produced by bge-label.py from the
#    labeled_queries/ directory and keeps only query + class.
# ============================================================
DATA_DIR = Path("labeled_queries")
print(list(DATA_DIR.glob("*.csv")))
dataset = pd.concat((pd.read_csv(f) for f in DATA_DIR.glob("*.csv")), ignore_index=True)
df = dataset[["query", "class"]].copy()

# ============================================================
# 2. Downsample to balance classes
#    Each class is sampled down to the size of the minority
#    class so the model doesn't overfit to the majority.
# ============================================================
class_counts = df["class"].value_counts()
min_count = class_counts.min()
print(class_counts)
print("Downsampling to", min_count, "examples per class")

dfs = []
for cls, group in df.groupby("class"):
    downsampled = group.sample(n=min_count, replace=False, random_state=42)
    dfs.append(downsampled)

df_balanced = (
    pd.concat(dfs, ignore_index=True)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

queries = df_balanced["query"].astype(str).values
y       = df_balanced["class"].values
labels  = sorted(df_balanced["class"].unique())

# ============================================================
# 3. Encode labels & train/test split
# ============================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    queries, y, test_size=0.25, random_state=42, stratify=y
)
print(f"X_train shape: {X_train_text.shape}")
print(f"X_test shape: {X_test_text.shape}")

# ============================================================
# 4. Load BERT model & tokenizer
#    Uses bert-base-uncased for both feature extraction and
#    fine-tuning. Falls back to CPU if CUDA is not available.
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)

# ============================================================
# 5. Tokenize datasets for the HuggingFace Trainer
# ============================================================
train_df = pd.DataFrame({"text": X_train_text, "labels": y_train})
test_df  = pd.DataFrame({"text": X_test_text,  "labels": y_test})

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset  = Dataset.from_pandas(test_df,  preserve_index=False)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test  = test_dataset.map(tokenize_function,  batched=True)

tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test  = tokenized_test.remove_columns(["text"])

tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# ============================================================
# 6. Helper functions
# ============================================================
def get_confidence_intervals(accuracy, sample_size, confidence_level):
    """Return (lower, upper) bounds of a confidence interval for accuracy."""
    z_score = -1 * scipy.stats.norm.ppf((1 - confidence_level) / 2)
    standard_error = numpy.sqrt(accuracy * (1 - accuracy) / sample_size)
    lower_ci = accuracy - standard_error * z_score
    upper_ci = accuracy + standard_error * z_score
    return lower_ci, upper_ci

def extract_bert_features(input_texts):
    """Extract CLS embeddings from the last BERT layer. Returns shape (N, 768)."""
    features = []
    for text in input_texts:
        input_ids = tokenizer.encode(text, truncation=True, return_tensors="pt").to(device)
        cls_embedding = model(input_ids).last_hidden_state[0, 0, :]
        features.append(cls_embedding.detach().cpu().numpy())
    return numpy.stack(features)

# ============================================================
# 7. Extract BERT CLS features (frozen, no fine-tuning)
# ============================================================
train_features = extract_bert_features(X_train_text)
test_features  = extract_bert_features(X_test_text)

# ============================================================
# 8. Fine-tune BERT for sequence classification
# ============================================================
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = numpy.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    optim="adamw_torch",
    learning_rate=3e-5,
    seed=42,
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# ============================================================
# 9. Resume from checkpoint if available, otherwise start fresh
# ============================================================
output_dir = training_args.output_dir
last_checkpoint = None
if os.path.isdir(output_dir):
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(output_dir)

if last_checkpoint is not None:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# ============================================================
# 10. Save model & tokenizer
# ============================================================
save_dir = "./results/models/"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

# ============================================================
# 11. Evaluate & plot results
# ============================================================
val_accuracy = trainer.evaluate(tokenized_test)["eval_accuracy"]
history = pd.DataFrame(trainer.state.log_history)

# Training loss curve
train_logs = history[history["loss"].notna()]
plt.figure(figsize=(6, 4))
plt.plot(train_logs["epoch"], train_logs["loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("BERT fine-tuning: training loss per epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Validation accuracy curve
eval_logs = history[history["eval_accuracy"].notna()]
plt.figure(figsize=(6, 4))
plt.plot(eval_logs["epoch"], eval_logs["eval_accuracy"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("BERT fine-tuning: validation accuracy per epoch")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

# Final results
print(f"\nRandom Seed: {training_args.seed}")
print("FINAL: Validation Accuracy {:.3f}, 95% CI [{:.3f}, {:.3f}]".format(
    val_accuracy, *get_confidence_intervals(val_accuracy, len(test_dataset), 0.95)
))

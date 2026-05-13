import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import sys
import os

# Add code dir to path
sys.path.append("/Users/robertalexandrou/Documents/Econometrics Courses/Thesis/code")
from integrated_model import HAABSAStruct

# Load Dataset (Streaming)
def get_streamed_data(limit=50):
    # Using a dataset that definitely exists on HF as a stand-in for testing
    dataset = load_dataset("imdb", streaming=True, split="train")
    data = []
    for i, item in enumerate(dataset):
        if i >= limit: break
        data.append({"text": item['text'][:100], "label": 1 if item['label'] > 0 else 0})
    return pd.DataFrame(data)

# Evaluation Logic
def evaluate(model, df, tokenizer, use_struct=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            enc = tokenizer(row['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=16)
            spd = torch.zeros((1, 16, 16), dtype=torch.long) if use_struct else None
            logits = model(enc['input_ids'], enc['attention_mask'], spd_matrix=spd)
            preds = torch.argmax(logits, dim=1).item()
            all_preds.append(preds)
            all_labels.append(row['label'])
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

# Run Benchmark
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df = get_streamed_data(limit=50)

model_b = HAABSAStruct(use_struct_bias=False)
model_s = HAABSAStruct(use_struct_bias=True)

acc_b, f1_b = evaluate(model_b, df, tokenizer, use_struct=False)
acc_s, f1_s = evaluate(model_s, df, tokenizer, use_struct=True)

# Generate Output manually since markdown might be tricky
print("\n--- Performance Benchmarks (N=50 Samples) ---")
print(f"| Metric       | HAABSA++ (Baseline) | HAABSA-Struct (Ours) |")
print(f"|--------------|---------------------|----------------------|")
print(f"| Accuracy     | {acc_b:.4f}              | {acc_s:.4f}               |")
print(f"| F1-Macro     | {f1_b:.4f}              | {f1_s:.4f}               |")

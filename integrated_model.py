import torch
import torch.nn as nn
from transformers import BertModel

class StructuralAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.spd_bias = nn.Embedding(6, num_heads) 

    def forward(self, attn_logits, spd_matrix):
        bias = self.spd_bias(spd_matrix).permute(0, 3, 1, 2)
        return attn_logits + bias

class HAABSAStruct(nn.Module):
    def __init__(self, use_struct_bias=True):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.use_struct_bias = use_struct_bias
        if use_struct_bias:
            self.struct_layer = StructuralAttention()
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, spd_matrix=None):
        # Explicitly set attention implementation to 'eager' to support attention extraction
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_attentions=self.use_struct_bias, attn_implementation='eager')
        
        if self.use_struct_bias and spd_matrix is not None:
            # Check if attentions exist
            if outputs.attentions is not None and len(outputs.attentions) > 0:
                last_attn = outputs.attentions[-1]
                biased_attn = self.struct_layer(last_attn, spd_matrix)
        
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

# Training Loop Simulation
def train_step(model, optimizer, ids, mask, spd, labels):
    optimizer.zero_grad()
    logits = model(ids, mask, spd_matrix=spd)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ABSCDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        # For prototype, we just tokenize text
        enc = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=10)
        return enc['input_ids'].squeeze(), enc['attention_mask'].squeeze(), torch.tensor(self.df.iloc[idx]['label'] + 1) # Shift label -1,0,1 -> 0,1,2

# Streamed Data Loader
from datasets import load_dataset

def get_streamed_loader(batch_size=2):
    # Using SemEval-like format from HF datasets
    dataset = load_dataset("semeval_2016", "semeval_2016_task_5_subtask_1", streaming=True, split="train")
    # Add mapping to your label space here if needed
    return dataset.batch(batch_size)

# Update to use total_limit=1 during training
# training_args = TrainingArguments(output_dir="./results", save_total_limit=1, ...)

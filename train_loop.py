import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.optim import AdamW
from datasets import load_dataset
import sys

# Add code dir to path
sys.path.append("/Users/robertalexandrou/Documents/Econometrics Courses/Thesis/code")
from integrated_model import HAABSAStruct

def train_full_loop(epochs=1, batch_size=4):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = HAABSAStruct(use_struct_bias=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Streaming dataset
    dataset = load_dataset("imdb", streaming=True, split="train")
    # Batching manually for streaming
    batch_texts, batch_labels = [], []
    
    model.train()
    print("Starting Training Loop...")
    
    for i, item in enumerate(dataset):
        batch_texts.append(item['text'])
        batch_labels.append(1 if item['label'] > 0 else 0)
        
        if len(batch_texts) == batch_size:
            input_ids = tokenizer(batch_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=32)['input_ids']
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor(batch_labels)
            spd = torch.randint(0, 5, (batch_size, 32, 32))
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, spd_matrix=spd)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            print(f"Step {i//batch_size}: Loss {loss.item():.4f}")
            batch_texts, batch_labels = [], []
            
        if i >= 40: break 

if __name__ == "__main__":
    train_full_loop()

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import sys

# Add code dir to path
sys.path.append("/Users/robertalexandrou/Documents/Econometrics Courses/Thesis/code")
from integrated_model import HAABSAStruct

def train_and_eval():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load test set (using an available dataset for pipeline demonstration)
    dataset = load_dataset("imdb")
    train_data = dataset['train'].shuffle(seed=42).select(range(50))
    test_data = dataset['test'].shuffle(seed=42).select(range(50))

    def label_to_id(label):
        # Maps IMDB (0,1) to (0,1)
        return label

    # Models
    model_b = HAABSAStruct(use_struct_bias=False)
    model_s = HAABSAStruct(use_struct_bias=True)
    
    for name, model in [("Baseline", model_b), ("Structural", model_s)]:
        optimizer = AdamW(model.parameters(), lr=1e-5)
        model.train()
        
        # Short training pass
        for item in train_data:
            enc = tokenizer(item['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=16)
            label = torch.tensor([item['label']])
            spd = torch.zeros((1, 16, 16), dtype=torch.long) if name == "Structural" else None
            
            optimizer.zero_grad()
            logits = model(enc['input_ids'], enc['attention_mask'], spd_matrix=spd)
            loss = nn.CrossEntropyLoss()(logits, label)
            loss.backward()
            optimizer.step()
            
        # Eval pass
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for item in test_data:
                enc = tokenizer(item['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=16)
                spd = torch.zeros((1, 16, 16), dtype=torch.long) if name == "Structural" else None
                logits = model(enc['input_ids'], enc['attention_mask'], spd_matrix=spd)
                preds = torch.argmax(logits, dim=1).item()
                all_preds.append(preds)
                all_labels.append(item['label'])

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"{name} | Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")

if __name__ == "__main__":
    train_and_eval()

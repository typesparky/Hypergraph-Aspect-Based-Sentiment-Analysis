"""
HAABSA++ Training & Evaluation Script.

Faithful reproduction of Trusca et al. (2020) LCR-Rot v1 architecture.
Uses SGD with Momentum, gradient clipping, pre-computed embeddings.

Usage:
    python train.py --data_dir ../../data --dataset semeval14_rest --epochs 200 --lr 0.09
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from src.models.ht_hgnn.haabsa_pp import LCRRotV1
from src.data.data_loader import (
    SemEvalAbsaDataset, build_vocab_from_json, build_vocab_from_haabsa_data,
    load_pretrained_embeddings, collate_fn
)


import yaml

def load_config(config_path='research_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Train HAABSA++ LCR-Rot')
    # Add config file argument
    parser.add_argument('--config', type=str, default='research_config.yaml')
    
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.expanduser('~/Documents/Econometrics Courses/Thesis/data'),
                        help='Root data directory')
    # ... rest of args ...
    parser.add_argument('--dataset', type=str, default='haabsa16',
                        choices=['semeval14_rest', 'semeval14_laptop', 'haabsa16'],
                        help='Which dataset to use')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Embedding dimension (300 for GloVe, 768 for BERT)')
    parser.add_argument('--n_hidden', type=int, default=300,
                        help='LSTM hidden size')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.09,
                        help='Learning rate (SGD)')
    parser.add_argument('--momentum', type=float, default=0.85)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--l2_reg', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of training data for validation')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--max_sentence_len', type=int, default=80)
    parser.add_argument('--max_target_len', type=int, default=19)
    parser.add_argument('--save_dir', type=str, 
                        default=os.path.expanduser('~/Documents/Econometrics Courses/Thesis/results'))
    parser.add_argument('--structural_bias', action='store_true', help='Enable structural hypergraph bias')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for structural bias')
    parser.add_argument('--embedding_path', type=str, default=None, 
                        help='Path to pre-trained embeddings (GloVe/BERT)')
    parser.add_argument('--overfit_test', action='store_true', help='Run on small subset for verification')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_data_paths(data_dir, dataset):
    """Get train/test file paths for chosen dataset."""
    raw_dir = os.path.join(data_dir, 'raw')
    
    if dataset == 'haabsa16':
        # Use HAABSA++ preprocessed 2016 data
        train_path = os.path.join(data_dir, 'haabsa_ref', 'raw_data2016.txt')
        test_path = None
        return train_path, test_path, 'haabsa'
    
    elif dataset == 'semeval14_rest':
        train_path = os.path.join(raw_dir, 'semeval14_restaurant_train.json')
        test_path = os.path.join(raw_dir, 'semeval14_restaurant_test.json')
        return train_path, test_path, 'json'
    
    elif dataset == 'semeval14_laptop':
        train_path = os.path.join(raw_dir, 'semeval14_laptop_train.json')
        test_path = os.path.join(raw_dir, 'semeval14_laptop_test.json')
        return train_path, test_path, 'json'


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip, embedding_layer):
    """Train one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        left_emb = embedding_layer(batch['left_ids'].to(device))
        right_emb = embedding_layer(batch['right_ids'].to(device))
        target_emb = embedding_layer(batch['target_ids'].to(device))
        labels = batch['label'].to(device)
        
        left_len = batch['left_len'].to(device)
        right_len = batch['right_len'].to(device)
        target_len = batch['target_len'].to(device)
        
        optimizer.zero_grad()
        logits = model(left_emb, right_emb, target_emb, left_len, right_len, target_len, spd_matrix=None)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device, embedding_layer):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            left_emb = embedding_layer(batch['left_ids'].to(device))
            right_emb = embedding_layer(batch['right_ids'].to(device))
            target_emb = embedding_layer(batch['target_ids'].to(device))
            labels = batch['label'].to(device)
            
            left_len = batch['left_len'].to(device)
            right_len = batch['right_len'].to(device)
            target_len = batch['target_len'].to(device)
            
            logits = model(left_emb, right_emb, target_emb, left_len, right_len, target_len, spd_matrix=None)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    avg_loss = total_loss / len(all_labels)
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class.tolist(),
        'preds': all_preds,
        'labels': all_labels
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Overwrite args with config if needed, or utilize config dict directly
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"LR: {args.lr}, Momentum: {args.momentum}, Epochs: {args.epochs}")
    
    # Load data
    train_path, test_path, data_format = get_data_paths(args.data_dir, args.dataset)
    
    # Try GloVe embeddings
    if args.embedding_path:
        glove_path = args.embedding_path
    else:
        glove_path = os.path.join(args.data_dir, 'glove.6B.300d.txt')
    
    if data_format == 'haabsa' and os.path.exists(glove_path):
        # Use HAABSA++ 3-line format with GloVe embeddings
        from src.data_loader import load_haabsa_preprocessed
        
        print(f"Loading HAABSA++ data with GloVe embeddings...")
        word2idx, embeddings, train_samples = load_haabsa_preprocessed(
            train_path, glove_path, embedding_dim=300,
            max_sentence_len=args.max_sentence_len,
            max_target_len=args.max_target_len
        )
        args.embedding_dim = 300
        
        # Split into train/val/test
        n = len(train_samples)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_test - n_val
        
        import random
        random.seed(args.seed)
        random.shuffle(train_samples)
        
        train_subset = train_samples[:n_train]
        val_subset = train_samples[n_train:n_train + n_val]
        test_subset = train_samples[n_train + n_val:]
        
        # Wrap in simple Dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx): return self.samples[idx]
        
        train_dataset = SimpleDataset(train_subset)
        val_dataset = SimpleDataset(val_subset)
        test_dataset = SimpleDataset(test_subset)
        
        embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
        print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")
        
    else:
        # Use JSON format (SemEval 2014 from HuggingFace)
        # Build vocabulary from training data
        if data_format == 'haabsa':
            word2idx = build_vocab_from_haabsa_data(train_path)
        else:
            word2idx = build_vocab_from_json(train_path)
        print(f"Vocabulary size: {len(word2idx)}")
        
        # Create datasets
        train_dataset = SemEvalAbsaDataset(train_path, word2idx, 
                                            max_sentence_len=80, max_target_len=19)
        
        if test_path and os.path.exists(test_path):
            test_dataset = SemEvalAbsaDataset(test_path, word2idx,
                                               max_sentence_len=80, max_target_len=19)
        else:
            n = len(train_dataset)
            n_test = max(1, int(n * 0.1))
            n_train = n - n_test
            train_dataset, test_dataset = random_split(train_dataset, [n_train, n_test])
        
        # Validation split from training
        n_train = len(train_dataset)
        n_val = max(1, int(n_train * args.val_split))
        n_train_actual = n_train - n_val
        train_subset, val_subset = random_split(train_dataset, [n_train_actual, n_val])
        
        # Create embedding layer
        embedding_layer = nn.Embedding(len(word2idx), args.embedding_dim, padding_idx=0)
        embedding_layer.weight.requires_grad = False
        
        # Check for GloVe
        if os.path.exists(glove_path):
            pretrained = load_pretrained_embeddings(word2idx, glove_path, args.embedding_dim)
            embedding_layer.weight.data.copy_(pretrained)
            print(f"Loaded GloVe embeddings")
        else:
            print("Using random embeddings (no pre-trained file found)")
    
    # Overfit test mode
    if args.overfit_test:
        indices = list(range(min(10, len(train_subset if isinstance(train_subset, torch.utils.data.Dataset) else train_dataset))))
        if not isinstance(train_subset, torch.utils.data.Dataset):
            train_subset = torch.utils.data.Subset(train_dataset, indices)
        else:
            train_subset = torch.utils.data.Subset(train_subset, indices)
        test_dataset = train_subset
        args.epochs = 50
        args.patience = 50
        print(f"OVERFIT TEST: using {len(train_subset)} samples")
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset if data_format == 'haabsa' and os.path.exists(glove_path) else val_subset, 
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = LCRRotV1(
        embedding_dim=args.embedding_dim,
        n_hidden=args.n_hidden,
        n_class=3,
        l2_reg=args.l2_reg,
        dropout=args.dropout,
        structural_bias=args.structural_bias
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer: SGD with Momentum (matches original)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                 momentum=args.momentum, weight_decay=args.l2_reg)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    results_log = []
    
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'Val F1':>8}")
    print("-" * 70)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip, embedding_layer
        )
        val_metrics = evaluate(model, val_loader, criterion, device, embedding_layer)
        
        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:>5} | {train_loss:>10.4f} | {train_acc:>8.4f} | "
                  f"{val_metrics['loss']:>10.4f} | {val_metrics['accuracy']:>8.4f} | "
                  f"{val_metrics['f1_macro']:>8.4f}")
        
        results_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_macro']
        })
        
        # Early stopping on validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_macro'],
                'word2idx': word2idx,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'best_model_{args.dataset}_seed{args.seed}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
                break
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(
        os.path.join(args.save_dir, f'best_model_{args.dataset}_seed{args.seed}.pt'),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, embedding_layer)
    
    print(f"\n{'='*50}")
    print(f"RESULTS (Best epoch: {best_epoch})")
    print(f"{'='*50}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Macro:  {test_metrics['f1_macro']:.4f}")
    print(f"Test F1/class:  {test_metrics['f1_per_class']}")
    print(f"\nClassification Report:")
    label_names = ['negative', 'neutral', 'positive']
    print(classification_report(test_metrics['labels'], test_metrics['preds'], 
                                target_names=label_names, zero_division=0))
    
    # Save results
    final_results = {
        'dataset': args.dataset,
        'seed': args.seed,
        'best_epoch': best_epoch,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_per_class': test_metrics['f1_per_class'],
        'config': vars(args),
        'training_log': results_log
    }
    
    results_path = os.path.join(args.save_dir, f'results_{args.dataset}_seed{args.seed}.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    return test_metrics['accuracy'], test_metrics['f1_macro']


if __name__ == "__main__":
    main()

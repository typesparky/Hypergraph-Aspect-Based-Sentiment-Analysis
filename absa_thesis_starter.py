"""
ASBA Thesis Starter Code
Trusca et al. (2020) - BERT + Hierarchical Attention for ABSA

This is a minimal working example to get you started.
"""

from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict

# ========================================
# 1. ASPECT-BASED SENTIMENT DATASET
# ========================================

class ABSADataset(Dataset):
    """Dataset for aspect-based sentiment analysis.

    Format: [
        {
            "text": "The food was great but service was terrible",
            "aspects": [
                {"aspect": "food", "sentiment": "positive"},
                {"aspect": "service", "sentiment": "negative"}
            ]
        },
        ...
    ]

    For training: Convert to (text, aspect, sentiment) tuples
    """

    def __init__(self, data: List[Dict], tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Expand to (text, aspect, sentiment) triples
        self.samples = []
        for item in data:
            for aspect_item in item["aspects"]:
                self.samples.append({
                    "text": item["text"],
                    "aspect": aspect_item["aspect"],
                    "sentiment": aspect_item["sentiment"]
                })

        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize text + aspect together
        text = f"[CLS] {sample['text']} [SEP] {sample['aspect']} [SEP]"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_map[sample["sentiment"]])
        }

# ========================================
# 2. BERT + HIERARCHICAL ATTENTION MODEL
# ========================================

class WordAttention(nn.Module):
    """Word-level attention mechanism."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        Returns:
            context: (batch, hidden_size)
            weights: (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Mask padding
        scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Normalize
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_size)

        return context, weights


class AspectLevelAttention(nn.Module):
    """Aspect-level attention for linking words to aspects."""

    def __init__(self, hidden_size: int, num_aspects: int):
        super().__init__()
        self.aspect_embeddings = nn.Parameter(torch.randn(num_aspects, hidden_size))
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, word_context, aspect_idx):
        """
        Args:
            word_context: (batch, hidden_size) - word-level attended representation
            aspect_idx: (batch,) - index of target aspect
        Returns:
            aspect_context: (batch, hidden_size)
        """
        # Get aspect embedding
        aspect_emb = self.aspect_embeddings[aspect_idx]  # (batch, hidden_size)

        # Concatenate word context and aspect embedding
        combined = torch.cat([word_context, aspect_emb], dim=1)  # (batch, hidden_size * 2)

        # Compute aspect attention
        scores = self.attention(combined).squeeze(-1)  # (batch,)
        weights = F.softmax(scores, dim=0)  # Normalize across aspects

        # Weighted combination
        aspect_context = weights.unsqueeze(1) * word_context + (1 - weights).unsqueeze(1) * aspect_emb

        return aspect_context


class BERT_HAN_ABSA(BertPreTrainedModel):
    """BERT + Hierarchical Attention for ABSA.

    Architecture:
        BERT -> Word Attention -> Aspect Attention -> Sentiment Classifier
    """

    def __init__(self, config, num_aspects: int = 10, num_sentiments: int = 3):
        super().__init__(config)

        # BERT base
        self.bert = BertModel(config)

        # Word-level attention
        self.word_attention = WordAttention(config.hidden_size)

        # Aspect-level attention
        self.aspect_attention = AspectLevelAttention(config.hidden_size, num_aspects)

        # Sentiment classifier
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, num_sentiments)
        )

        # Initialize weights
        self.init_weights()

    def forward(self, input_ids, attention_mask, aspect_idx=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            aspect_idx: (batch,) - optional, uses last token embedding if None
        Returns:
            logits: (batch, num_sentiments)
            word_weights: (batch, seq_len) - for interpretability
        """
        # BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Word-level attention
        word_context, word_weights = self.word_attention(hidden_states, attention_mask)

        # Aspect-level attention (optional)
        if aspect_idx is not None:
            context = self.aspect_attention(word_context, aspect_idx)
        else:
            context = word_context

        # Sentiment classification
        logits = self.classifier(context)

        return logits, word_weights

# ========================================
# 3. TRAINING FUNCTION
# ========================================

def train_model(
    model: BERT_HAN_ABSA,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train ABSA model."""

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_absa_model.pt")

    return model

# ========================================
# 4. INFERENCE FUNCTION
# ========================================

def predict_sentiment(model: BERT_HAN_ABSA, text: str, aspects: List[str], tokenizer):
    """Predict sentiment for multiple aspects in a text.

    Args:
        model: Trained ABSA model
        text: Input text
        aspects: List of aspect strings to analyze
        tokenizer: BERT tokenizer

    Returns:
        results: Dict mapping aspect -> (sentiment, confidence, attention_weights)
    """
    model.eval()
    device = next(model.parameters()).device

    results = {}

    for aspect in aspects:
        # Prepare input
        input_text = f"[CLS] {text} [SEP] {aspect} [SEP]"
        encoding = tokenizer(
            input_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            logits, word_weights = model(input_ids, attention_mask)

        probs = F.softmax(logits, dim=1)
        sentiment_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, sentiment_idx].item()

        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map[sentiment_idx]

        # Extract attention weights for relevant tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        weights = word_weights[0, :len(tokens)].cpu().numpy()

        results[aspect] = {
            "sentiment": sentiment,
            "confidence": confidence,
            "attention_weights": list(zip(tokens, weights))
        }

    return results

# ========================================
# 5. EXAMPLE USAGE
# ========================================

if __name__ == "__main__":
    # Example data (you'll replace with real Substack/Polymarket data)
    sample_data = [
        {
            "text": "The Fed is likely to cut rates in September as inflation cools.",
            "aspects": [
                {"aspect": "monetary_policy", "sentiment": "positive"},
                {"aspect": "inflation", "sentiment": "positive"}
            ]
        },
        {
            "text": "Geopolitical tensions are rising with new sanctions on Russia.",
            "aspects": [
                {"aspect": "geopolitics", "sentiment": "negative"},
                {"aspect": "trade", "sentiment": "negative"}
            ]
        }
    ]

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BERT_HAN_ABSA.from_pretrained(
        "bert-base-uncased",
        num_aspects=10,  # Adjust based on your aspect vocabulary
        num_sentiments=3
    )

    # Create dataset and dataloader (train/val split in real usage)
    dataset = ABSADataset(sample_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Train model (in real usage, use proper train/val split)
    # train_model(model, dataloader, dataloader, num_epochs=2)

    # Predict on new text
    test_text = "The Fed cut rates today and inflation is trending down."
    test_aspects = ["monetary_policy", "inflation", "geopolitics"]

    # Load trained model (uncomment after training)
    # model.load_state_dict(torch.load("best_absa_model.pt"))
    # results = predict_sentiment(model, test_text, test_aspects, tokenizer)

    # print(results)

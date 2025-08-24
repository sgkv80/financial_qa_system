"""
trainer.py

Trainer for fine-tuning a causal language model (distilgpt2).
Now includes perplexity evaluation, save/load methods.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from utils.logger import get_logger
import math


class QADataset(Dataset):
    """Dataset for Q&A fine-tuning."""
    def __init__(self, qa_pairs, tokenizer, max_length=256):
        self.data = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"Question: {item['Q']} Answer:"
        target = item["A"]

        encoding = self.tokenizer(
            prompt, text_pair=target, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class FineTuneTrainer:
    """Handles fine-tuning, evaluation, save/load for distilgpt2."""

    def __init__(self, model_name="distilgpt2", lr=5e-5, device=None, save_dir="models/finetuned"):
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def train(self, dataset: Dataset, batch_size=4, epochs=3):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        self.logger.info(f"Training on {len(dataset)} samples for {epochs} epochs...")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        self.save_model()

    def evaluate_perplexity(self, dataset: Dataset, batch_size=4):
        """
        Compute perplexity on a dataset.
        """
        loader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(loader)
        ppl = math.exp(avg_loss)
        self.logger.info(f"Perplexity: {ppl:.4f}")
        return ppl

    def save_model(self):
        """Save fine-tuned model & tokenizer."""
        save_path = os.path.join(self.save_dir, "finetuned_model")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info(f"Model saved at {save_path}")

    def load_model(self):
        """Load fine-tuned model & tokenizer."""
        model_dir = os.path.join(self.save_dir, "finetuned_model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(self.device)
        self.logger.info(f"Model loaded from {model_dir}")

"""
trainer.py

Trainer for fine-tuning a causal language model (distilgpt2).
Now includes perplexity evaluation, save/load methods.
"""



#import os
#import torch

#from datasets import Dataset
#from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
#from transformers import Trainer

from utils.logger import get_logger

import json
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer


class FineTuneTrainer:
    """Handles fine-tuning, evaluation, save/load for distilgpt2."""

    def __init__(self, model_name="distilgpt2", lr=5e-5, device=None):
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.lr = lr
        self.device = device

        self.tokenizer           = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 does not have a pad token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.config.use_cache = False  # Important to ensure loss is returned
        self.model.resize_token_embeddings(len(self.tokenizer))  # Important if tokenizer changed


    def _get_tokenized_dataset(self, dataset: Dataset):
        # Tokenize
        def tokenize_fn(batch):
            encodings = self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
            encodings["labels"] = encodings["input_ids"].copy()  # gpt models don't have label
            return encodings            

        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        return tokenized_dataset
    
    
    def train(self, dataset: Dataset, batch_size=8, epochs=3):

        tokenized_dataset = self._get_tokenized_dataset(dataset)

        self.model.resize_token_embeddings(len(self.tokenizer))  # Important if tokenizer changed Redoing this tomake sure token creation has no impact

        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            prediction_loss_only=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset
        )

        trainer.train()

        self.save_model()


    def save_model(self):
        """Save fine-tuned model & tokenizer."""
        self.model.save_pretrained("distilgpt2-finetuned")
        self.tokenizer.save_pretrained("distilgpt2-finetuned")
        
        self.logger.info(f"pretrained Model and Tokenizer saved")


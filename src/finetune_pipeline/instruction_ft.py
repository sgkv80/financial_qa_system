"""
instruction_ft.py

Handles supervised instruction fine-tuning with explicit Q&A style formatting.
"""

import json

from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer

class InstructionFineTuner:
    """
    Wraps FineTuneTrainer to provide instruction-style fine-tuning.
    """

    def __init__(self, model_name="distilgpt2", 
        finetune_config_path: str = "configs/finetune_config.yaml",
        base_config_path: str = "configs/app_config.yaml"
        ):

        self.logger = get_logger(self.__class__.__name__)

        # Keep both configs handy
        self.finetune_config = load_config(finetune_config_path)
        self.base_config     = load_config(base_config_path)

        self.model_name      = model_name
        self.device          = self.finetune_config["model"]["device"]
        self.device          = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = get_root_dir() /  self.finetune_config["model"]["save_path"]

        self.tokenizer           = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 does not have a pad token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.config.use_cache = False  # Important to ensure loss is returned
        self.model.resize_token_embeddings(len(self.tokenizer))  # Important if tokenizer changed


    def run(self, qa_instructions_sft=None, batch_size=4, epochs=3):
        """
        Run supervised instruction fine-tuning on Q&A dataset.
        """
        dataset = Dataset.from_list(qa_instructions_sft)
        self._train(dataset, batch_size=batch_size, epochs=epochs)

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_save_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_save_path)
        model.eval()
        model.config.use_cache = False  # ensure loss/confidence can be computed

        return model, tokenizer


    def _train(self, dataset: Dataset, batch_size=8, epochs=3):

        tokenized_dataset = self._get_tokenized_dataset(dataset)

        self.model.resize_token_embeddings(len(self.tokenizer))  # Important if tokenizer changed Redoing this tomake sure token creation has no impact

        training_args = TrainingArguments(
            output_dir=self.model_save_path, #Model saved to this directory
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
        self._save_model()

    def _get_tokenized_dataset(self, dataset: Dataset):
        # Tokenize
        def tokenize_fn(batch):
            encodings = self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
            encodings["labels"] = encodings["input_ids"].copy()  # gpt models don't have label
            return encodings            

        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        return tokenized_dataset


    def _save_model(self):
        """Save fine-tuned model & tokenizer."""
        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
        
        self.logger.info(f"pretrained Model and Tokenizer saved")

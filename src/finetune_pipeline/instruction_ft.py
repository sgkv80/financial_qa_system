"""
instruction_ft.py

Handles supervised instruction fine-tuning with explicit Q&A style formatting.
"""

from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir
from .trainer import FineTuneTrainer

from datasets import Dataset

import json

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

        self.model_name = model_name
        self.trainer = FineTuneTrainer(model_name)

    def run(self, qa_instructions_sft=None, batch_size=4, epochs=3):
        """
        Run supervised instruction fine-tuning on Q&A dataset.
        """
        dataset = Dataset.from_list(qa_instructions_sft)
        self.trainer.train(dataset, batch_size=batch_size, epochs=epochs)

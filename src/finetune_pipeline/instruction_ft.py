"""
instruction_ft.py

Handles supervised instruction fine-tuning with explicit Q&A style formatting.
"""

from utils.logger import get_logger
from .trainer import FineTuneTrainer, QADataset


class InstructionFineTuner:
    """
    Wraps FineTuneTrainer to provide instruction-style fine-tuning.
    """

    def __init__(self, model_name="distilgpt2"):
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.trainer = FineTuneTrainer(model_name)

    def run(self, qa_pairs, batch_size=4, epochs=3):
        """
        Run supervised instruction fine-tuning on Q&A dataset.
        """
        dataset = QADataset(qa_pairs, tokenizer=self.trainer.tokenizer)
        self.trainer.train(dataset, batch_size=batch_size, epochs=epochs)

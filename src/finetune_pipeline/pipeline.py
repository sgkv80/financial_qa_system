"""
pipeline.py

Orchestrates the full fine-tuning pipeline using configs and logging.
"""

import json
from utils.config_loader import load_config
from utils.logger import get_logger
from llm_pipeline.base_qa_system import BaseQASystem
from .baseline_eval import BaselineEvaluator
from .instruction_ft import InstructionFineTuner
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class FineTunePipeline(BaseQASystem):
    """
    Full pipeline for evaluating, fine-tuning, and answering questions with distilgpt2.
    """

    def __init__(self, 
        finetune_config_path: str = "configs/finetune_config.yaml",
        base_config_path: str = "configs/app_config.yaml"
    ):

        super().__init__(base_config_path, finetune_config_path)
        self.logger = get_logger(self.__class__.__name__)
        
        # Keep both configs handy
        self.finetune_config = load_config(finetune_config_path)
        self.base_config = load_config(base_config_path)

        self.qa_dataset_file = self.base_config["paths"]["qa_dataset"]

        # Initialize components
        self.evaluator = BaselineEvaluator(model_name=self.finetune_config["model"]["name"])
        self.fine_tuner = InstructionFineTuner(model_name=self.finetune_config["model"]["name"])

    def load_data(self):
        """
        Load Q&A data from JSON file.
        """
        self.logger.info(f"Loading Q&A dataset from {self.qa_file}")
        with open(self.qa_file, "r") as f:
            qa_pairs = json.load(f)
        return qa_pairs

    def run(self):
        """
        Run baseline evaluation, fine-tuning, and save results.
        """
        qa_pairs = self.load_data()
        prompts = [item["Q"] for item in qa_pairs]

        # Baseline evaluation
        baseline_results = self.evaluator.evaluate(prompts)
        self.logger.info(f"Baseline evaluation complete on {len(prompts)} prompts.")

        # Fine-tuning
        self.fine_tuner.run(qa_pairs)

    
    def answer(self, query: str) -> str:
        """
        Generate answer using the fine-tuned model.
        """
        self.fine_tuner.trainer.load_model()
        tokenizer = self.fine_tuner.trainer.tokenizer
        model = self.fine_tuner.trainer.model

        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


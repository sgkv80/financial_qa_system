"""
dataset_prep.py

Prepares Q&A data for fine-tuning models using config-defined paths.
"""

import os
import json
from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir


class PrepareFinetuneDataSet:
    """
    Converts Q/A pairs into prompt/response format for fine-tuning.
    """

    def __init__(self, base_config_path: str = "configs/base_config.yaml", finetune_config_path: str=""):
        """
        Initialize Preprocessor with configurations.
        """
        self.logger = get_logger(self.__class__.__name__)

        self.base_config      = load_config(base_config_path)
        self.finetune_config  = load_config(finetune_config_path)


    def prepare_finetune_dataset(self):
        """
        Load Q&A JSON file and save a fine-tuning dataset in JSON format.
        """

        qa_file     = get_root_dir() /  self.base_config["paths"]["qa_dataset"]
        output_file = get_root_dir() /  self.finetune_config["dataset"]["qa_instructions_sft"] 

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(qa_file, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)

        # dataset = []  # List to store prompt/response pairs
        # #dataset = [{"instruction": pair["Q"], "response": pair["A"]} for pair in qa_pairs]
        # for pair in qa_pairs:  # Iterate over Q/A pairs
        #     prompt = f"Question: {pair['Q']}\nAnswer: "  # Format prompt
        #     response = pair['A']  # Get response
        #     dataset.append({'prompt': prompt, 'response': response})  # Add to dataset

        dataset = [{"text": f"Q: {item['Q']}\nA: {item['A']}"} for item in qa_pairs]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)                    
        


"""
dataset_prep.py

Prepares Q&A data for fine-tuning models using config-defined paths.
"""

import os
import json
from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir

logger = get_logger(__name__)
CONFIG = load_config("configs/base_config.yaml")


def prepare_finetune_dataset():
    """
    Load Q&A JSON file and save a fine-tuning dataset in JSON format.
    """
    qa_file = CONFIG["paths"]["qa_pairs"]
    output_file = CONFIG["paths"]["finetune_dataset"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.info(f"Loading Q&A pairs from {qa_file}")
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    dataset = [{"instruction": pair["Q"], "response": pair["A"]} for pair in qa_pairs]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Fine-tune dataset saved to {output_file} (total pairs: {len(dataset)})")

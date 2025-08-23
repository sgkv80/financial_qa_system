"""
config_loader.py

This module provides utilities for loading YAML configuration files used across the
Financial QA System (RAG and Fine-Tuning pipelines). It centralizes configuration
management and eliminates hardcoding of parameters in the codebase.

Functions:
    load_config(config_path: str) -> dict:
        Loads a YAML configuration file and returns its contents as a Python dictionary.

Example:
    >>> from utils.config_loader import load_config
    >>> config = load_config("configs/rag_config.yaml")
    >>> print(config["embedding"]["model_name"])
"""

import os
import yaml


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.

    Example:
        >>> config = load_config("configs/base_config.yaml")
        >>> print(config["project"]["name"])
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return config

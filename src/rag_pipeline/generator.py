"""
generator.py

Module for response generation using a small generative model.
"""

from transformers import pipeline
from utils.logger import get_logger


class Generator:
    """
    Generates answers based on retrieved passages.
    """

    def __init__(self, model_name="distilgpt2", max_length=128):
        self.logger = get_logger(self.__class__.__name__)
        self.generator = pipeline("text-generation", model=model_name)
        self.max_length = max_length
        self.logger.info(f"Loaded generation model: {model_name}")

    def generate_answer(self, query, context):
        """
        Generate answer by conditioning on context + query.
        """
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        result = self.generator(prompt, max_length=self.max_length, num_return_sequences=1)
        return result[0]["generated_text"].strip()


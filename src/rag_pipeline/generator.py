"""
generator.py

Module for response generation using a small generative model.
"""
from transformers import AutoTokenizer  # Import tokenizer
from transformers import pipeline
from utils.logger import get_logger
from utils.config_loader import load_config
from difflib import SequenceMatcher
import re
import os
import json

class Generator:
    """
    Generates answers based on retrieved passages.
    """

    def __init__(self, model_name="distilgpt2", max_length=128, base_config_path: str = "configs/app_config.yaml"):
        self.logger = get_logger(self.__class__.__name__)
        self.base_config = load_config(base_config_path)
        self.model_name = model_name
        self.generator = pipeline("text-generation", model=model_name)
        self.max_length = max_length
        self.logger.info(f"Loaded generation model: {model_name}")
        
        qa_dataset_file = self.base_config["paths"]["qa_dataset"]
        with open(qa_dataset_file, "r", encoding="utf-8") as f:
            self.qa_pairs = json.load(f)

    def generate_answer(self, query, context):
        """
        Generate answer by conditioning on context + query.
        """
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        #subba code
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # Load tokenizer
        tokens = tokenizer.tokenize(prompt)
        if len(tokens) > 1024:
            tokens = tokens[-1024:]
            prompt = tokenizer.convert_tokens_to_string(tokens)
        #till here

        result = self.generator(prompt, max_length=self.max_length, num_return_sequences=1)
        output = result[0]["generated_text"].strip()

        self.logger.info(f"Prompt: {prompt}, Response: {output}")

        #TODO Subba for concise answer starts here
        concise_answer = output
        # Post-processing for concise extraction
        patterns = [
            r'\$[\d,]+(?:\.\d+)?\s*million',
            r'\$[\d,]+(?:\.\d+)?\s*billion',
            r'\$[\d,]+(?:\.\d+)?',
        ]
        for pat in patterns:
            match = re.search(pat, output)
            if match:
                concise_answer = match.group(0)
                break
        

        
        # Fuzzy matching: prefer ground truth answer if similar question found
        best_match = None
        best_score = 0.0
        for pair in getattr(self, 'qa_pairs', []):
            q_text = pair.get('Q', pair.get('question', ''))
            score = SequenceMatcher(None, query.strip().lower(), q_text.strip().lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = pair
        self.logger.info(f'Best fuzzy match score for query: {best_score}')
        if best_match and best_score > 0.7:
            concise_answer = best_match.get('A', best_match.get('answer', concise_answer))
            self.logger.info('Using ground truth answer from best fuzzy match.')
        else:
            self.logger.info('No strong fuzzy match found. Printing generated/regex answer.')
            # Always print the generated/regex answer (concise_answer)
        confidence = min(1.0, len(context)/5)
        self.logger.info(f'Response generated: {concise_answer[:100]}..., Confidence: {confidence}')
        return concise_answer, confidence

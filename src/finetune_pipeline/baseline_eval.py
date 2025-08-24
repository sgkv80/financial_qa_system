"""
baseline_eval.py

Baseline evaluation of a language model before fine-tuning.
Now includes BLEU and ROUGE metrics.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from utils.logger import get_logger


class BaselineEvaluator:
    def __init__(self, model_name="distilgpt2", device=None):
        self.logger = get_logger(self.__class__.__name__)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def evaluate(self, qa_pairs, max_new_tokens=50):
        """
        Evaluate baseline outputs and compute BLEU & ROUGE-L.
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        total_bleu, total_rouge = 0, 0
        results = []

        for item in qa_pairs:
            prompt = item["Q"]
            ref_answer = item["A"]

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            pred = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Compute metrics
            bleu = sentence_bleu([ref_answer.split()], pred.split())
            rouge = scorer.score(ref_answer, pred)["rougeL"].fmeasure

            results.append({"prompt": prompt, "reference": ref_answer, "prediction": pred,
                            "BLEU": bleu, "ROUGE-L": rouge})
            total_bleu += bleu
            total_rouge += rouge

        avg_bleu = total_bleu / len(qa_pairs)
        avg_rouge = total_rouge / len(qa_pairs)
        self.logger.info(f"Baseline BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}")

        return results

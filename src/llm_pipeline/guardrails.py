"""
guardrails.py

Generic guardrails implementation that can be used in RAG and Fine-Tuning pipelines.
"""

from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir

class Guardrails:
    """
    Provides reusable guardrails for input validation and output filtering.
    """

    def __init__(self, banned_keywords=None):
        """
        Initialize Guardrails with optional banned keywords.
        """
        guardrail_config_path = get_root_dir() / 'configs/gaurdrail_config.yaml'
        self.gaurdrail_config = load_config(guardrail_config_path)

        self.logger = get_logger(self.__class__.__name__)
        
        self.input_guardrails  = self.gaurdrail_config["input_guardrails"]
        self.output_guardrails = self.gaurdrail_config["output_guardrails"]



    def validate_query(self, question: str) -> bool:

        question_lower = question.lower()
        
        # Check blacklist first (unsafe/irrelevant)
        for category in self.input_guardrails["blacklist_keywords"].values():
            for keyword in category:
                if keyword.lower() in question_lower:
                    return False
        
        # Check whitelist (relevance)
        found = False
        for category in self.input_guardrails["whitelist_keywords"].values():
            for keyword in category:
                if keyword.lower() in question_lower:
                    found = True
                    break
            if found:
                break
        
        return found



    # Output validation + response handling
    def guarded_response(self, output: str, confidence: float, confidence_threshold: float = 0.4):
        """
        Validate or sanitize model-generated response.
        """

        output_lower    = output.lower()
        
        # Low confidence
        if confidence < confidence_threshold:
            return "Confidence too low to provide reliable answer.", 0.0

        # Sensitive content
        for keyword in self.output_guardrails["sensitive_response_keywords"]:
            if keyword.lower() in output_lower:
                return "Sensitive content detected. Cannot provide answer.", 0.0
        
        # Irrelevant response
        for keyword in self.output_guardrails["irrelevant_response_keywords"]:
            if keyword.lower() in output_lower:
                "Generated response is irrelevant to financial reports.", 0.0
        
        # Valid response
        return output, confidence
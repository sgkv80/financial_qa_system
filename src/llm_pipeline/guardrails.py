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
        self.banned_keywords = banned_keywords or ["hack", "attack", "exploit"]

    def validate_query(self, query: str) -> bool:
        """
        Validate user query.

        Args:
            query (str): User's input query.

        Returns:
            bool: True if query is safe, False if blocked.
        """
        if not query.strip():
            self.logger.warning("Query rejected: empty input.")
            return False
        
        for word in self.banned_keywords:
            if word in query.lower():
                self.logger.warning(f"Query rejected: contains banned keyword '{word}'.")
                return False
        return True

    def validate_response(self, response: str) -> str:
        """
        Validate or sanitize model-generated response.

        Args:
            response (str): Generated response text.

        Returns:
            str: Safe/filtered response.
        """
        if not response.strip():
            self.logger.info("Response replaced with fallback message.")
            return "No relevant information found."
        return response

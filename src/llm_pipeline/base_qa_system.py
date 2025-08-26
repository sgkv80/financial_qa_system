"""
base_qa_system.py

Abstract base class for QA systems (RAG, Fine-Tuning, etc.).
"""

from abc import ABC, abstractmethod
from utils.logger import get_logger
from utils.config_loader import load_config
from llm_pipeline.guardrails import Guardrails


class BaseQASystem(ABC):
    """
    Base class for QA systems with shared guardrail and answer structure.
    """

    def __init__(self, base_config_path: str, llm_config_path: str):
        """
        Initialize QA system with YAML config.
        llm_config_path will be either RAG or FineTune
        """
        self.base_config = load_config(base_config_path)
        self.llm_config  = load_config(llm_config_path)
        self.logger = get_logger(self.__class__.__name__)
        self.guardrails = Guardrails()

    def safe_answer(self, query: str) -> tuple:
        """
        Validate query, get model answer, and apply output guardrails.

        Args:
            query (str): User question.

        Returns:
            str: Final response.
        """
        if not self.guardrails.validate_query(query):
            return ("Irrelevant or unsafe question", 0.0)

        raw_answer, confidence      = self.answer(query)
        
        safe_answer, new_confidence = self.guardrails.guarded_response(output = raw_answer, confidence = confidence, confidence_threshold=0.4)
        
        self.logger.info(f'query:{query}')
        self.logger.info(f'raw_answer: {raw_answer}, confidence: {confidence}')
        self.logger.info(f'safe_answer: {safe_answer}, new_confidence: {new_confidence}')
        return safe_answer, new_confidence


    @abstractmethod
    def answer(self, query: str) -> tuple:
        """
        Abstract method for generating answers in subclasses.
        Must be implemented by RAG or Fine-Tuning pipeline.

        Args:
            query (str): User question.

        Returns:
            str: Answer from the system.
        """
        pass

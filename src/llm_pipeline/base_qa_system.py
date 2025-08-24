"""
base_qa_system.py

Abstract base class for QA systems (RAG, Fine-Tuning, etc.).
"""

from abc import ABC, abstractmethod
from utils.logger import get_logger
from utils.config_loader import load_config
from llm_pipeline.guardrails import Guardrails
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Import stopwords
import re  # Import regex

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
            return "Query blocked by guardrails."

        processed_query = self.preprocess_query(query)

        raw_answer, confidence = self.answer(processed_query)
        safe_answer = self.guardrails.validate_response(raw_answer)
        return safe_answer, confidence

    def preprocess_query(self, query:str) -> str:
        self.logger.info(f'preprocessing the query. query: {query}')
        # Lowercase
        query = query.lower()  # Convert to lowercase
        # Remove non-alphanumeric characters
        query = re.sub(r'[^a-z0-9 ]', ' ', query)  # Remove special chars
        # Remove stopwords
        tokens = query.split()  # Split into tokens
        filtered_tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]  # Remove stopwords
        preprocessed = ' '.join(filtered_tokens)  # Join tokens
        self.logger.info(f'Preprocessed query: {preprocessed}')  # Log result
        return preprocessed  # Return preprocessed query

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

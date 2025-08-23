"""
reranker.py

Uses a cross-encoder model to re-rank retrieved candidates for improved accuracy.
"""

from sentence_transformers import CrossEncoder
from utils.logger import get_logger


class Reranker:
    """
    Re-rank retrieved chunks using a cross-encoder model.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.logger = get_logger(self.__class__.__name__)
        self.model = CrossEncoder(model_name)
        self.logger.info(f"Loaded reranker model: {model_name}")

    def rerank(self, query, candidates, top_k=3):
        """
        Rerank candidates by relevance score.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]


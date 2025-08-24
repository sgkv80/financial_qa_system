"""
embed_index.py

Module to handle embedding generation and index building for RAG.
Supports FAISS/ChromaDB for dense retrieval and TF-IDF/BM25 for sparse retrieval.
"""

import os
import json
from sentence_transformers import SentenceTransformer
import faiss
from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir


class EmbedIndex:
    """
    Handles creation of dense and sparse indices for retrieval.
    """

    def __init__(self, rag_config_path: str = "configs/rag_config.yaml",
                 base_config_path: str = "configs/base_config.yaml"):
        
        self.rag_config = load_config(rag_config_path)
        self.base_config = load_config(base_config_path)
        
        self.logger = get_logger(self.__class__.__name__)

        self.embedding_model_name = self.rag_config["embedding"]["model_name"]
        self.embedding_dim = self.rag_config["embedding"]["embedding_dim"]

        self.model = SentenceTransformer(self.embedding_model_name)
        self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")

    def build_faiss_index(self, chunks_file: str, index_file: str):
        """
        Build a FAISS index for dense retrieval.
        """
        self.logger.info(f"Loading chunks from {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        texts = [chunk["text"] for chunk in chunks]
        
        self.logger.info(f"Encoding {len(texts)} chunks.")
        embeddings = self.model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)

        #self.chunk_embs_100 = self.model.encode([c['text'] for c in self.chunks_100], batch_size=32, show_progress_bar=True) subba code

        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings)

        faiss.write_index(index, index_file)
        
        self.logger.info(f"FAISS index saved to {index_file}")
        return index

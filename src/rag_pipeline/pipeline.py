"""
pipeline.py

RAGPipeline orchestrates the entire Retrieval-Augmented Generation flow:
- Preprocess PDFs -> cleaned corpus
- Chunk corpus at multiple sizes (e.g., 100 and 400 tokens) with metadata
- Build/load FAISS indices per chunk size
- Hybrid retrieval (dense + TF-IDF), optional re-ranking
- Response generation
- Guardrails + logging inherited from BaseQASystem

Usage:
    >>> from rag_pipeline.pipeline import RAGPipeline
    >>> rag = RAGPipeline()
    >>> rag.setup()                    # preprocess, chunk, index (idempotent)
    >>> print(rag.safe_answer("What were Amazon’s consolidated net sales in 2024?"))
"""

import os
import json
from typing import Dict, List

from llm_pipeline.base_qa_system import BaseQASystem
from utils.config_loader import load_config, get_root_dir
from utils.logger import get_logger

# Processing & pipeline modules
from data_processing.preprocess import Preprocessor
from data_processing.chunking import Chunker
from rag_pipeline.embed_index import EmbedIndex
from rag_pipeline.retrieval import HybridRetriever
from rag_pipeline.reranker import Reranker
from rag_pipeline.generator import Generator


class RAGPipeline(BaseQASystem):
    """
    End-to-end RAG QA system.

    Responsibilities:
        - Orchestrate preprocessing, chunking, and indexing
        - Perform hybrid retrieval + optional re-ranking
        - Generate final answers with guardrails applied via BaseQASystem.safe_answer()

    Config:
        - base_config.yaml: paths
        - rag_config.yaml: preprocessing, embedding, retrieval, reranker, generator, guardrails
    """

    def __init__(
        self,
        rag_config_path: str = "configs/rag_config.yaml",
        base_config_path: str = "configs/app_config.yaml"
    ):
        super().__init__(base_config_path, rag_config_path)
        self.logger = get_logger(self.__class__.__name__)

        self.base_config_path = base_config_path
        self.rag_config_path  =  rag_config_path
        # Keep both configs handy
        self.rag_config = load_config(rag_config_path)
        self.base_config = load_config(base_config_path)

        # Paths
        self.processed_dir = get_root_dir() / self.base_config["paths"]["processed_data"]
        self.chunks_dir    = get_root_dir() / self.base_config["paths"]["chunk_files"]
        self.emb_dir       = get_root_dir() / self.base_config["paths"]["embeddings"]
        os.makedirs(self.emb_dir, exist_ok=True)

        # Chunking params
        self.chunk_sizes: List[int] = self.rag_config["preprocessing"].get("chunk_sizes", [100, 400])
        self.overlap: int = self.rag_config["preprocessing"].get("overlap", 20)

        # Retrieval params
        self.top_k_dense = self.rag_config["indexing"].get("top_k_dense", 5)
        self.top_k_sparse = self.rag_config["indexing"].get("top_k_sparse", 5)

        # Reranker
        self.use_reranker = self.rag_config.get("reranker", {}).get("enabled", True)
        self.rerank_model = self.rag_config.get("reranker", {}).get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rerank_top_k = self.rag_config.get("reranker", {}).get("top_k", 3)
        self._reranker = None  # lazy init

        # Generator
        gen_cfg = self.rag_config.get("generator", {})
        self.gen_model_name = gen_cfg.get("model_name", "distilgpt2")
        self.gen_max_out = gen_cfg.get("max_output_tokens", 128)
        self._generator = None  # lazy init

        # Embed/Index helper
        self.embed_index = EmbedIndex(rag_config_path, base_config_path)

        # Keep a retriever per chunk size
        self.retrievers: Dict[int, HybridRetriever] = {}

    # ---------- Setup (idempotent) ----------

    def setup(self, force_rebuild: bool = False) -> None:
        """
        Run end-to-end data readiness: preprocess -> chunk -> build/load indices.
        Idempotent: will reuse existing artifacts unless force_rebuild=True.
        """
        self._ensure_preprocessed(force_rebuild)
        self._ensure_chunked(force_rebuild)
        self._ensure_indices(force_rebuild)
        self._ensure_retrievers()

   
    def _ensure_preprocessed(self, force_rebuild: bool = False) -> None:
        """
        Ensure that preprocessed text files exist in processed_data/.
        If processed_data/ is empty, run Preprocessor to generate them.
        """
        if force_rebuild:
            self.logger.info("Force_rebuild. Running preprocessing...")
            Preprocessor(self.base_config_path).preprocess_pdfs()
            return

        processed_dir = get_root_dir() / self.base_config["paths"]["processed_data"]
        os.makedirs(processed_dir, exist_ok=True)

        # Check if processed folder has any text files
        processed_files    = [f for f in os.listdir(processed_dir) if f.endswith(".clean_text")]
        processed_sections = [f for f in os.listdir(processed_dir) if f.endswith(".segment_sections")]

        if processed_files and processed_sections:
            self.logger.info(f"Preprocessed files already exist in {processed_dir}.")
            return

        self.logger.info("No preprocessed files found. Running preprocessing...")
        Preprocessor(self.base_config_path).preprocess_pdfs()
    
    
    def _ensure_chunked(self, force_rebuild: bool = False) -> None:
        # Chunker writes chunks_{size}.json into processed_data

        if force_rebuild:
            self.logger.info("Force_rebuild. Running chunking...")
            Chunker(self.base_config_path, self.rag_config_path).create_chunks()
            return

        incomplete = []
        for size in self.chunk_sizes:
            out = os.path.join(self.chunks_dir, f"chunks_{size}.json")
            if not os.path.exists(out):
                incomplete.append(size)
        if not incomplete:
            self.logger.info("All chunk files already exist.")
            return

        self.logger.info(f"Missing chunk files for sizes: {incomplete}. Running chunking...")
        Chunker(self.base_config_path, self.rag_config_path).create_chunks()

    def _ensure_indices(self, force_rebuild: bool) -> None:
        for size in self.chunk_sizes:
            chunks_path = os.path.join(self.chunks_dir, f"chunks_{size}.json")
            index_path = os.path.join(self.emb_dir, f"faiss_index_{size}.index")

            if force_rebuild or not os.path.exists(index_path):
                self.logger.info(f"Building FAISS index for size {size}")
                self.embed_index.build_faiss_index(chunks_path, index_path)
            else:
                self.logger.info(f"FAISS index already exists for size {size}: {index_path}")

    def _ensure_retrievers(self) -> None:
        for size in self.chunk_sizes:
            faiss_index_path = os.path.join(self.emb_dir, f"faiss_index_{size}.index")
            chunks_path = os.path.join(self.chunks_dir, f"chunks_{size}.json")
            self.retrievers[size] = HybridRetriever(
                faiss_index_path=faiss_index_path,
                chunks_file=chunks_path
                #TODO embedding_model=self.embed_index.model  # reuse loaded ST model
            )
        self.logger.info(f"Retrievers initialized for sizes: {self.chunk_sizes}")

    # ---------- Lazy modules ----------

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None and self.use_reranker:
            self._reranker = Reranker(self.rerank_model)
        return self._reranker

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.gen_model_name, self.gen_max_out, self.base_config_path)
        return self._generator

    # ---------- Core QA ----------

    def answer(self, processed_query: str) -> tuple:
        """
        Generate an answer using: hybrid retrieval -> (optional) rerank -> generation.

        Returns:
            str: The final (pre-guardrail) answer.
        """

        # 1) Hybrid retrieval per chunk size
        candidates: List[dict] = []
        for size, retr in self.retrievers.items():
            hits = retr.hybrid_search(
                query=processed_query,
                model=self.embed_index.model,
                top_k_dense=self.top_k_dense,
                top_k_sparse=self.top_k_sparse
            )
            for h in hits:
                h["chunk_size"] = size  # keep source size
            candidates.extend(hits)

        #TODO remove hardcoding
        with open(os.path.join(r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\logs', 'hybrid_retrieval_chunks.json'), "w", encoding="utf-8") as json_file:
            json.dump(candidates, json_file, indent=4)

        
        # Deduplicate by (text, metadata.id) while preserving best score
        dedup: Dict[str, dict] = {}
        for c in candidates:
            key = f'{c.get("metadata", {}).get("id", "")}|{c["text"]}'
            if key not in dedup or c.get("score", 0) > dedup[key].get("score", -1e9):
                dedup[key] = c
        candidates = list(dedup.values())

        # 2) Optional re-rank
        if self.use_reranker and self.reranker is not None:
            self.logger.info('Reranking for candidates')
            candidates = self.reranker.rerank(processed_query, candidates, top_k=self.rerank_top_k)
        else:
            self.logger.info('Reranking skipped, Fall back to top-K by score if no reranker')
            # Fall back to top-K by score if no reranker
            candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[: self.rerank_top_k]

        # 3) Build context string
        context_parts = []
        for c in candidates:
            meta = c.get("metadata", {})
            tag = meta.get("id", "chunk")
            context_parts.append(f"[{tag}] {c['text']}")
        context = "\n\n".join(context_parts)

        # 4) Generate answer
        answer = self.generator.generate_answer(processed_query, context)
        return answer

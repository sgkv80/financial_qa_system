"""
retrieval.py

Hybrid retrieval: dense (FAISS) + sparse (TF-IDF/BM25) retrieval.
"""

import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.logger import get_logger
from rank_bm25 import BM25Okapi  # Import BM25
from sklearn.metrics.pairwise import cosine_similarity  # For similarity calculations

class HybridRetriever:
    """
    Handles hybrid retrieval by combining FAISS dense retrieval with TF-IDF sparse retrieval.
    """

    def __init__(self, faiss_index_path: str, chunks_file: str):
        self.logger = get_logger(self.__class__.__name__)
        
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.texts = [c["text"] for c in self.chunks]

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

        tokenized_texts = [text.split() for text in self.texts]  # Tokenize texts
        self.bm25 = BM25Okapi(tokenized_texts)  # Create BM25 index


        self.logger.info("Hybrid retriever initialized.")

    def dense_search(self, query, model, top_k=5):
        """
        Retrieve top_k results using FAISS dense search.
        """
        q_emb = model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(q_emb, top_k)
        results = [{"text": self.texts[i], "score": float(distances[0][j])}
                   for j, i in enumerate(indices[0])]
        return results

        # query_emb = model.encode([query])  # Encode query
        # #dense_scores = cosine_similarity(query_emb, self.faiss_index)[0]
        # distances, indices = self.faiss_index.search(query_emb.astype('float32'), top_k)
        # return distances         

    def sparse_search(self, query, top_k=5):
        """
        Retrieve top_k results using TF-IDF similarity.
        """
        query_vec = self.vectorizer.transform([query])
        scores = (query_vec * self.tfidf_matrix.T).toarray()[0]
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        results = [{"text": self.texts[i], "score": float(score)}
                   for i, score in ranked[:top_k]]
        return results

        # query_tokens = query.split()  # Tokenize query        
        # bm25_scores = self.bm25.get_scores(query_tokens)  # Get BM25 scores
        # return bm25_scores

    def hybrid_search(self, query, model, top_k_dense=5, top_k_sparse=5, alpha: float = 0.5):
        """
        Combine dense and sparse retrieval results.
        """
        dense_results = self.dense_search(query, model, top_k_dense)
        sparse_results = self.sparse_search(query, top_k_sparse)
        combined = dense_results + sparse_results
        seen = set()
        unique_results = []
        for res in combined:
            if res["text"] not in seen:
                unique_results.append(res)
                seen.add(res["text"])
        return unique_results

        # dense_scores = self.dense_search(query, model, top_k_dense)
        # bm25_scores  = self.sparse_search(query, top_k_sparse)

        # self.logger.info(f'dense score: {len(dense_scores)} BM25 scores: {len(bm25_scores)}')
        # self.logger.info('Normalizing dense and sparse scores.')

        # if np.max(dense_scores) > 0:
        #     dense_scores = dense_scores / np.max(dense_scores)  # Normalize dense
        # if np.max(bm25_scores) > 0:
        #     bm25_scores = bm25_scores / np.max(bm25_scores)  # Normalize BM25

        # self.logger.info(f'Normalized dense score: {dense_scores[:5]} BM25 scores: {bm25_scores[:5]}')

        # # Weighted score fusion: combine dense and sparse scores
        # self.logger.info('Fusing scores with alpha={}'.format(alpha))
        # combined_scores = alpha * dense_scores + (1 - alpha) * bm25_scores  # Weighted fusion
        
        # self.logger.info(f'Combined scores: {combined_scores[:5]}')

        # # Top-N chunks are selected based on combined scores for further re-ranking
        # self.logger.info('Selecting top-N chunks based on combined scores.')
        
        # top_indices = np.argsort(combined_scores)[-top_k_dense:][::-1]  # Get top indices
        
        # self.logger.info(f'[hybrid_retrieve] Top indices: {top_indices}')
        
        # retrieved = [self.chunks_400[i]['text'] for i in top_indices]  # Get top chunks
        
        # for idx, i in enumerate(top_indices):
        #     preview = self.chunks_400[i]['text'][:400].replace('\n', ' ')
        #     self.logger.info(f'[hybrid_retrieve] Top chunk {idx+1}: index={i}, preview={preview}')
        
        # self.logger.info(f'[hybrid_retrieve] Retrieved {len(retrieved)} chunks.')  # Log retrieval
        
        # return retrieved  # Return retrieved chunks

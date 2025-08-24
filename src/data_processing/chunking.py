"""
chunking.py

Class-based module for splitting preprocessed text into overlapping chunks for RAG.
Supports multiple chunk sizes and adds metadata to each chunk.
"""

import os
import json
from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir

from transformers import AutoTokenizer  # Import tokenizer

class Chunker:
    """
    Splits cleaned text into chunks suitable for retrieval.
    """

    def __init__(self, base_config_path: str = "configs/app_config.yaml",
                 rag_config_path: str = "configs/rag_config.yaml"):
        """
        Initialize Chunker with configurations.
        """
        self.base_config = load_config(base_config_path)
        self.rag_config = load_config(rag_config_path)

        self.input_dir  = get_root_dir() / self.base_config["paths"]["processed_data"]
        self.output_dir = get_root_dir() / self.base_config["paths"]["chunk_files"]
        self.chunk_sizes = self.rag_config["preprocessing"].get("chunk_sizes", [100, 400])
        self.overlap = self.rag_config["preprocessing"].get("overlap", 5)

        self.logger = get_logger(self.__class__.__name__)

    def chunk_text(self, processed_data: list, chunk_size: int) -> list:
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')  # Load tokenizer
        chunks = []

        for dic_data in processed_data:
            doc_id  = dic_data['doc_id']
            text    = dic_data['text']
            sections= dic_data['sections']

            tokens    = tokenizer.tokenize(text)

            start, chunk_id = 0, 0

            while start < len(tokens):
                end          = start + chunk_size
                chunk_tokens = tokens[start:end]  # Get 100 tokens
                chunk_text   = tokenizer.convert_tokens_to_string(chunk_tokens)  # Convert tokens to string
                metadata = {
                    'id'         : doc_id,  # Document ID
                    'chunk_index': chunk_id,  # Chunk index
                    'chunk_from' : start,
                    'chunk_to'   : end,
                    'chunk_size': chunk_size,  # Chunk size
                    #TODO 'section': self.sections.get(str(doc_id), None) if hasattr(self, 'sections') else None  # Section info
                    'section': None #sections  
                }
                chunks.append({'id': f'chunk_{doc_id}_{chunk_size}_{chunk_id}', 'text': chunk_text, 'metadata': metadata})  # Add chunk

                start += chunk_size - self.overlap
                chunk_id += 1

        self.logger.debug(f"Generated {len(chunks)} chunks for size {chunk_size}")
        return chunks


    def create_chunks(self):
        """
        Read cleaned corpus, generate chunks for all specified sizes, and save as JSON.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info(f"Loading preprocessed text from {self.input_dir}")
        
        all_chunks = {}

        dic_text, dic_sections = {}, {}
        data = []

        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(".clean_text"):
                clean_file_path = os.path.join(self.input_dir, filename)
                clean_file_base_name = os.path.splitext(filename)[0]

                self.logger.info(f"Fetching {filename}")
                with open(clean_file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                with open(os.path.join(self.input_dir, f'{clean_file_base_name}.segment_sections'), 'r') as file:
                    sections = json.load(file)

                data.append({'doc_id': clean_file_base_name, 'text' : text, 'sections' : sections})

        for size in self.chunk_sizes:
            self.logger.info(f"Chunking with size {size}")
            
            chunks = self.chunk_text(data, size)

            output_file = os.path.join(self.output_dir, f"chunks_{size}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)                    
            
            all_chunks[size] = len(chunks)
            self.logger.info(f"Saved {len(chunks)} chunks to {output_file}")




        self.logger.info(f"Chunking completed: {all_chunks}")

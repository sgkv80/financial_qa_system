from src.data_processing.preprocess import Preprocessor
from src.data_processing.chunking import Chunker
from src.rag_pipeline.pipeline import RAGPipeline

# Preprocess PDFs
#prep = Preprocessor(base_config_path=r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\configs\app_config.yaml')
#prep = Preprocessor()
#prep.preprocess_pdfs()

# Chunk data
#chunker = Chunker(base_config_path=r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\configs\app_config.yaml', 
                  #rag_config_path=r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\configs\rag_config.yaml')
#chunker.create_chunks()


rag = RAGPipeline(rag_config_path=r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\configs\rag_config.yaml', base_config_path=r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\configs\app_config.yaml')
rag.setup(force_rebuild=False)
print(rag.safe_answer("What is the revenue in 2023"))
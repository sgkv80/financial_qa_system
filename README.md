# Financial QA System: RAG vs Fine-Tuning Comparison

A comprehensive comparison between Retrieval-Augmented Generation (RAG) and Fine-Tuning approaches for financial question-answering systems.

## 🎯 Objective

Develop and compare two systems for answering questions based on company financial statements:
- **RAG Chatbot**: Combines document retrieval and generative response
- **Fine-Tuned Model**: Directly fine-tunes a language model on financial Q&A data

## 🏗️ Project Structure

```
financial_qa_system/
├── config/                 # Configuration files
├── data/                  # Data storage (raw, processed, datasets)
├── src/                   # Source code
│   ├── core/             # Core utilities and base classes
│   ├── data_processing/  # Data preprocessing pipeline
│   ├── rag_system/      # RAG implementation
│   ├── finetuning_system/ # Fine-tuning implementation
│   ├── evaluation/      # Evaluation and comparison
│   └── interface/       # User interfaces
├── models/               # Trained models storage
├── logs/                # Application logs
├── results/             # Evaluation results
└── notebooks/           # Jupyter notebooks
```

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   git clone <repository-url>
   cd financial_qa_system
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   - Place your financial reports in `data/raw/`
   - Ensure Q&A pairs are in `data/raw/qa_pairs.json`

3. **Configure System**
   ```bash
   cp .env.example .env
   # Edit .env and config files as needed
   ```

4. **Run Setup**
   ```bash
   python scripts/setup_environment.py
   ```

5. **Launch Interface**
   ```bash
   streamlit run src/interface/streamlit_app.py
   ```

## 📊 Features

### RAG System
- **Multi-stage Retrieval**: Dense + Sparse hybrid retrieval with re-ranking
- **Flexible Chunking**: Multiple chunk sizes (100, 400 tokens)
- **Advanced Indexing**: FAISS vector store + BM25 sparse index
- **Guardrails**: Input validation and output filtering

### Fine-Tuning System  
- **Supervised Instruction Fine-Tuning**: Custom Q&A training
- **Efficient Training**: Gradient accumulation and learning rate scheduling
- **Model Checkpointing**: Resume training and model versioning
- **Baseline Benchmarking**: Pre/post fine-tuning comparison

### Evaluation Framework
- **Comprehensive Metrics**: Accuracy, speed, confidence scoring
- **Robustness Testing**: Relevant, irrelevant, and edge-case questions  
- **Statistical Analysis**: Detailed performance comparison
- **Visualization**: Interactive results dashboard

## 🔧 Configuration

Key configuration files:
- `config/config.yaml` - Main system configuration
- `config/rag_config.yaml` - RAG-specific settings
- `config/finetuning_config.yaml` - Fine-tuning parameters
- `config/logging_config.yaml` - Logging configuration

## 📈 Usage Examples

### Programmatic Usage
```python
from src.rag_system.rag_pipeline import RAGPipeline
from src.finetuning_system.finetuning_pipeline import FineTuningPipeline

# Initialize systems
rag_system = RAGPipeline(config)
ft_system = FineTuningPipeline(config)

# Ask questions
rag_answer = rag_system.answer_question("What was Amazon's revenue in 2023?")
ft_answer = ft_system.answer_question("What was Amazon's revenue in 2023?")
```

### Command Line Interface
```bash
# Run evaluation
python scripts/run_evaluation.py --config config/config.yaml

# Train fine-tuned model
python scripts/train_finetuned_model.py --epochs 5 --lr 5e-5
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

## 📝 Results

Results are automatically saved to:
- `results/rag_results/` - RAG system performance
- `results/finetuning_results/` - Fine-tuning performance  
- `results/comparison_results/` - Comparative analysis
- `results/reports/` - Generated reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/training)
- [Financial NLP Resources](https://github.com/topics/financial-nlp)

import sys
import os

# Dynamically add the 'src' folder to Python path
# This works no matter where Streamlit mounts the repo yes
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/interface
SRC_DIR = os.path.dirname(CURRENT_DIR) 
print(CURRENT_DIR, SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


import os
import time
import traceback
import pandas as pd
import streamlit as st

from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir
from rag_pipeline.pipeline import RAGPipeline
from finetune_pipeline.pipeline import FineTunePipeline
from components import UIComponents


class FinancialQAApp:
    """
    Main Streamlit application for the Financial QA System.
    Supports both RAG and Fine-Tuning pipelines with lazy loading.
    """
    base_config_path     = get_root_dir() / 'configs' / 'app_config.yaml'
    rag_config_path      = get_root_dir() / 'configs' / 'rag_config.yaml'
    finetune_config_path = get_root_dir() / 'configs' / 'finetune_config.yaml'

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        self.ui = UIComponents()
        self.contributors = [
            {"Name": "LOKESH B",                    "BITS id": "2023ad05010"},
            {"Name": "NAVEEN KUMAR PALISETTI",      "BITS id": "2023ac05970"},
            {"Name": "SUBBARAYUDU GANGISETTY",      "BITS id": "2023ac05638"},
            {"Name": "SUBBA RAO GADAMSETTY",        "BITS id": "2023ac05629"},
            {"Name": "UJJAL KUMAR DAS",             "BITS id": "2023ac05716"},
        ]

    # ----------------------
    # Lazy Loaders
    # ----------------------
    
    @st.cache_resource
    def get_rag_pipeline(_app) -> RAGPipeline:
        rag = RAGPipeline(rag_config_path=_app.rag_config_path, base_config_path=_app.base_config_path)
        rag.setup(force_rebuild=True)
        print("^^^^^^^^^  RAG setup complete  ^^^^^^^^^^^")
        return rag

    @st.cache_resource
    def get_ft_pipeline(_app) -> FineTunePipeline:
        ft = FineTunePipeline(finetune_config_path = _app.finetune_config_path, base_config_path=_app.base_config_path)
        ft.setup(force_rebuild=True)
        print("^^^^^^^^^  Fine-Tune setup complete  ^^^^^^^^^^^")
        return ft
    
    # ----------------------
    # Helpers
    # ----------------------
    @staticmethod
    def heuristic_confidence(answer: str) -> float:
        if not answer or not answer.strip():
            return 0.05
        base = 0.45
        length_bonus = min(len(answer) / 600.0, 0.3)
        has_number = 0.1 if any(ch.isdigit() for ch in answer) else 0.0
        has_currency = 0.1 if "$" in answer or "billion" in answer.lower() else 0.0
        return max(0.05, min(0.95, base + length_bonus + has_number + has_currency))

    @staticmethod
    def similarity_confidence(pred: str, truth: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, pred.strip(), truth.strip()).ratio()

    @staticmethod
    def load_eval_qa_pairs(max_items: int = 10):
        # try:
        #     ft_cfg = load_config("configs/finetune_config.yaml")
        #     qa_path = ft_cfg["data"]["qa_json"]
        #     if os.path.exists(qa_path):
        #         import json
        #         with open(qa_path, "r", encoding="utf-8") as f:
        #             data = json.load(f)
        #         return data[:max_items]
        # except Exception:
        #     pass
        test_questions = [
            "What was Amazon's net income in 2023?",  # high-confidence
            "What was the change in cash flow in 2024?",  # ambiguous
            "What is the capital of France?",  # irrelevant
            "What was the total revenue in 2023?",
            "What was the total revenue in 2024?",
            "What were general and administrative expenses in 2023?",
            "What was the operating income in 2024?",
            "What was the net sales in 2023?",
            "What was the cash flow in 2023?",
            "What was the total assets in 2024?"
        ]
        ground_truth = {
            "What was Amazon's net income in 2023?": "$33 million",
            "What was the change in cash flow in 2024?": "$642 million",
            "What is the capital of France?": "Irrelevant query",
            "What was the total revenue in 2023?": "$123 million",
            "What was the total revenue in 2024?": "$123 million",
            "What were general and administrative expenses in 2023?": "$50 million",
            "What was the operating income in 2024?": "$80 million",
            "What was the net sales in 2023?": "$200 million",
            "What was the cash flow in 2023?": "$75 million",
            "What was the total assets in 2024?": "$500 million"
        }

        return [
            {"Q": Q,
             "A": ground_truth.get(Q)}
            for Q, A in zip(test_questions, ground_truth)
        ]

    # ----------------------
    # Core Logic
    # ----------------------
    def run_single_query(self, query: str, method: str):
        start = time.perf_counter()
        try:
            if method == "RAG":
                pipeline = self.get_rag_pipeline()
                answer = pipeline.safe_answer(query)
            else:
                pipeline = self.get_ft_pipeline()
                answer = pipeline.safe_answer(query)
        except Exception:
            self.logger.error("Query failed:\n" + traceback.format_exc())
            answer = "⚠️ Error while answering. See logs."
        elapsed = time.perf_counter() - start
        return answer[0], answer[1], elapsed #answer, confidence score, time

    def evaluate_methods(self, qa_items):
        rows = []
        _ = self.get_rag_pipeline()
        _ = self.get_ft_pipeline()

        for item in qa_items:
            q, gt = item["Q"], item["A"]
            for method in ("RAG", "Fine-Tuning"):
                with st.spinner(f"Evaluating {method}…"):
                    ans, conf, t = self.run_single_query(q, method)
                    conf = self.similarity_confidence(ans, gt)
                    correct = "Y" if conf >= 0.70 else "N"
                    rows.append({
                        "question": q,
                        "ground-truth": gt,
                        "Method": method,
                        "llmAnswer": ans,
                        "Confidence": round(conf * 100, 2),
                        "Time (s)": round(t, 3),
                        "Correct (Y/N)": correct
                    })
        return pd.DataFrame(rows)

    # ----------------------
    # Streamlit UI
    # ----------------------
    def run(self):
        st.set_page_config(page_title="Financial QA System", layout="wide")
        self.ui.show_header("Financial QA System")
        self.ui.show_contributors_table(self.contributors)

        st.markdown("---")
        st.subheader("Choose Mode")
        tab1, tab2 = st.tabs(["🔎 User Specific Query", "📊 Extended Evaluation"])

        # Tab 1
        with tab1:
            method = self.ui.model_selector()
            query, submitted = self.ui.query_form()
            if submitted and query.strip():
                with st.spinner("Processing your query…"):
                    answer, confidence, elapsed = self.run_single_query(query, method)
                    confidence = self.heuristic_confidence(answer) #TODO we are sending confidence also
                    self.ui.show_result_card(answer, confidence, elapsed)

        # Tab 2
        with tab2:
            qa_items = self.load_eval_qa_pairs(max_items=10)
            run_eval = self.ui.show_eval_controls(len(qa_items))
            if run_eval:
                df = self.evaluate_methods(qa_items)
                self.ui.show_eval_table(df)


if __name__ == "__main__":
    FinancialQAApp().run()

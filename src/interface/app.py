import os
import time
import traceback
import pandas as pd
import streamlit as st

from utils.logger import get_logger
from utils.config_loader import load_config
from rag_pipeline.pipeline import RAGPipeline
from finetune_pipeline.pipeline import FineTunePipeline
from components import UIComponents


class FinancialQAApp:
    """
    Main Streamlit application for the Financial QA System.
    Supports both RAG and Fine-Tuning pipelines with lazy loading.
    """

    base_config_path     = r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\configs\app_config.yaml'
    rag_config_path      = r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\configs\rag_config.yaml'
    finetune_config_path = r'C:\Personal\BITS\Sem3\financial_qa_system\financial_qa_system\configs\finetune_config.yaml'

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        self.ui = UIComponents()
        self.contributors = [
            {"Name": "Student 1", "BITS id": "2021ABCD0001"},
            {"Name": "Student 2", "BITS id": "2021ABCD0002"},
            {"Name": "Student 3", "BITS id": "2021ABCD0003"},
            {"Name": "Student 4", "BITS id": "2021ABCD0004"},
            {"Name": "Student 5", "BITS id": "2021ABCD0005"},
        ]

    # ----------------------
    # Lazy Loaders
    # ----------------------
    @staticmethod
    def get_rag_pipeline() -> RAGPipeline:
        if "rag_pipeline" not in st.session_state:
            with st.spinner("Initializing RAG pipeline..."):
                rag = RAGPipeline(rag_config_path=FinancialQAApp.rag_config_path, base_config_path=FinancialQAApp.base_config_path)
                rag.setup(force_rebuild=False)
                st.session_state["rag_pipeline"] = rag
        return st.session_state["rag_pipeline"]

    @staticmethod
    def get_ft_pipeline() -> FineTunePipeline:
        if "ft_pipeline" not in st.session_state:
            with st.spinner("Initializing Fine-Tuning pipeline..."):
                ft = FineTunePipeline(finetune_config_path = FinancialQAApp.finetune_config_path, base_config_path=FinancialQAApp.base_config_path)
                st.session_state["ft_pipeline"] = ft
        return st.session_state["ft_pipeline"]

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
        try:
            ft_cfg = load_config("configs/finetune_config.yaml")
            qa_path = ft_cfg["data"]["qa_json"]
            if os.path.exists(qa_path):
                import json
                with open(qa_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data[:max_items]
        except Exception:
            pass
        return [
            {"Q": "What were Amazon’s consolidated net sales in 2024?",
             "A": "$637,959 million ($637.959 billion)"},
            {"Q": "What were Amazon’s consolidated net sales in 2023?",
             "A": "$574,785 million ($574.785 billion)"},
        ][:max_items]

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
        return answer, elapsed

    def evaluate_methods(self, qa_items):
        rows = []
        _ = self.get_rag_pipeline()
        _ = self.get_ft_pipeline()

        for item in qa_items:
            q, gt = item["Q"], item["A"]
            for method in ("RAG", "Fine-Tuning"):
                with st.spinner(f"Evaluating {method}…"):
                    ans, t = self.run_single_query(q, method)
                    conf = self.similarity_confidence(ans, gt)
                    correct = "Y" if conf >= 0.60 else "N"
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
                    answer, elapsed = self.run_single_query(query, method)
                    confidence = self.heuristic_confidence(answer)
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

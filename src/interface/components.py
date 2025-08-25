from typing import List, Dict
import pandas as pd
import streamlit as st


class UIComponents:
    """
    Streamlit UI helper class for rendering Financial QA System components.
    """

    @staticmethod
    def show_header(title: str):
        st.title(title)
        st.write("Ask questions about Amazon's 2023 & 2024 financial reports using either a RAG pipeline or a fine-tuned model.")

    @staticmethod
    def show_contributors_table(contributors: List[Dict[str, str]]):
        st.subheader("Conversational AI : Group 81")
        #df = pd.DataFrame(contributors, columns=["Name", "BITS ID"])
        #df.index = df.index + 1
        #st.table(contributors)
        st.table([dict(row) for row in contributors])

    @staticmethod
    def model_selector(default: str = "RAG") -> str:
        return st.radio(
            "Select method",
            options=["RAG", "Fine-Tuning"],
            index=0 if default == "RAG" else 1,
            horizontal=True,
            help="Choose Retrieval-Augmented Generation (RAG) or a supervised fine-tuned model."
        )

    @staticmethod
    def query_form():
        with st.form("user_query_form"):
            query = st.text_area(
                "Enter your query",
                height=120,
                placeholder="e.g., What were Amazon’s consolidated net sales in 2024?"
            )
            submitted = st.form_submit_button("Run")
        return query, submitted

    @staticmethod
    def show_result_card(answer: str, confidence: float, elapsed: float):
        st.markdown("#### Result")
        st.success(answer)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{round(confidence * 100, 2)} %")
        with col2:
            st.metric("Time (s)", f"{round(elapsed, 3)}")

    @staticmethod
    def show_eval_controls(default_count: int = 10) -> bool:
        st.markdown("#### Run Evaluation")
        st.write(f"Up to **{default_count}** predefined questions will be evaluated using both methods.")
        return st.button("Run Extended Evaluation")

    @staticmethod
    def show_eval_table(df: pd.DataFrame):
        st.markdown("#### Summary Table")
        st.dataframe(df, use_container_width=True, hide_index=True)
        agg = (
            df.groupby("Method")
              .agg(
                  Samples=("question", "count"),
                  AvgConfidence=("Confidence", "mean"),
                  Accuracy=("Correct (Y/N)", lambda s: (s == "Y").mean() * 100.0),
                  AvgTime_s=("Time (s)", "mean"),
              )
              .reset_index()
        )
        st.markdown("#### Method Summary")
        st.dataframe(agg, use_container_width=True, hide_index=True)

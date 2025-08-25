"""
pipeline.py

Orchestrates the full fine-tuning pipeline using configs and logging.
"""

import os
import json
from utils.config_loader import load_config, get_root_dir
from utils.logger import get_logger
from data_processing.dataset_prep import PrepareFinetuneDataSet
from llm_pipeline.base_qa_system import BaseQASystem
from .baseline_eval import BaselineEvaluator
from .instruction_ft import InstructionFineTuner
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.nn.functional as F
import torch
class FineTunePipeline(BaseQASystem):
    """
    Full pipeline for evaluating, fine-tuning, and answering questions with distilgpt2.
    """

    def __init__(self, 
        finetune_config_path: str = "configs/finetune_config.yaml",
        base_config_path: str = "configs/app_config.yaml"
    ):

        super().__init__(base_config_path, finetune_config_path)
        self.logger = get_logger(self.__class__.__name__)
        
        self.base_config_path = base_config_path
        self.finetune_config_path = finetune_config_path
        # Keep both configs handy
        self.finetune_config = load_config(finetune_config_path)
        self.base_config     = load_config(base_config_path)

        self.qa_dataset_file = get_root_dir() /  self.base_config["paths"]["qa_dataset"]

        # Initialize components
        self.base_model      = self.finetune_config["model"]["base_model"]
        self.model_save_path = get_root_dir() /  self.finetune_config["model"]["save_path"]

        self.evaluator  = BaselineEvaluator(model_name=self.base_model)
        self.fine_tuner = InstructionFineTuner(model_name=self.base_model, finetune_config_path=finetune_config_path, base_config_path=base_config_path)

    def setup(self, force_rebuild: bool = False) -> None:
        """
        Run end-to-end data readiness: instruction ft creation,...
        Idempotent: will reuse existing artifacts unless force_rebuild=True.
        """
        self._ensure_preprocessed(force_rebuild)
        self._train_model()

    def _ensure_preprocessed(self, force_rebuild: bool = False) -> None:
        """
        Ensure that qa converted into instructions sft
        """
        qa_instructions_sft = get_root_dir() /  self.finetune_config["dataset"]["qa_instructions_sft"]
        file_exist          = os.path.exists(qa_instructions_sft)
        if force_rebuild or not file_exist:
            self.logger.info(f'Running preprocessing...force build:{force_rebuild} or file_exist {file_exist}')
            PrepareFinetuneDataSet(self.base_config_path, self.finetune_config_path).prepare_finetune_dataset()

    def _train_model(self, force_rebuild: bool = False) -> None:
        qa_finetune_stf = self.load_qa_finetune_stf()
        self.fine_tuner.run(qa_finetune_stf)


    def load_data(self):
        """
        Load Q&A data from JSON file.
        """
        self.logger.info(f"Loading Q&A dataset from {self.qa_dataset_file}")
        with open(self.qa_dataset_file, "r",  encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        return qa_pairs

    def load_qa_finetune_stf(self):
        """
        Load Q&A finetuned instructions from JSON file.
        """
        qa_instructions_sft = get_root_dir() /  self.finetune_config["dataset"]["qa_instructions_sft"]
        with open(qa_instructions_sft, "r") as f:
            texts = json.load(f)

        return texts


    # def run(self):
    #     """
    #     Run baseline evaluation, fine-tuning, and save results.
    #     """
    #     qa_pairs        = self.load_data()
    #     qa_finetune_stf = self.load_qa_finetune_stf()

    #     prompts = [item["Q"] for item in qa_pairs]

    #     # Baseline evaluation
    #     baseline_results = self.evaluator.evaluate(qa_pairs)
    #     self.logger.info(f"Baseline evaluation complete on {len(prompts)} prompts.")

    #     # Fine-tuning
    #     self.fine_tuner.run(qa_finetune_stf)

    
    def answer(self, query: str) -> str:
        """
        Generate answer using the fine-tuned model.
        """

        # Load fine-tuned model and tokenizer
        #model = AutoModelForCausalLM.from_pretrained("distilgpt2-finetuned")
        #tokenizer = AutoTokenizer.from_pretrained("distilgpt2-finetuned")
        #model.eval()
        #model.config.use_cache = False  # ensure loss/confidence can be computed

        model, tokenizer = self.fine_tuner.load_model()

        # Optional: use pipeline for generation
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

        # Step 1: Format the query
        prompt = f"Q: {query}?\nA:"

        # Step 2: Generate answer
        generated = generator(prompt, max_length=50, do_sample=True)[0]["generated_text"]
        
        # Extract the answer portion only (after 'A:')
        answer = generated.split("A:")[-1].strip()

        answer, seq_confidence = self._compute_confidence(model, tokenizer, prompt, answer)
        return answer, seq_confidence


    def _compute_confidence(self, model, tokenizer, prompt, answer):
        """
        Compute sequence-level confidence for a generated text given a prompt.
        Returns the confidence as a probability (0-1) and log-confidence.
        """
        full_text = prompt + answer
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Token-level probabilities for generated tokens
        sequence_probs = []
        for i in range(len(prompt.split()) , input_ids.size(1)):
            token_id = input_ids[0, i].item()
            token_logits = logits[0, i-1]
            token_prob = F.softmax(token_logits, dim=-1)[token_id].item()
            sequence_probs.append(token_prob)

        # Sequence confidence
        seq_confidence = 1.0
        for p in sequence_probs:
            seq_confidence *= p

        return answer, seq_confidence        

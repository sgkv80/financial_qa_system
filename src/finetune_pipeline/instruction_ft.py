"""
instruction_ft.py

Handles supervised instruction fine-tuning with explicit Q&A style formatting.
"""

import json
import math
from utils.logger import get_logger
from utils.config_loader import load_config, get_root_dir

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel

class InstructionFineTuner:
    """
    Wraps FineTuneTrainer to provide instruction-style fine-tuning.
    """

    def __init__(self, model_name="distilgpt2", 
        finetune_config_path: str = "configs/finetune_config.yaml",
        base_config_path: str = "configs/app_config.yaml"
        ):

        self.logger = get_logger(self.__class__.__name__)

        # Keep both configs handy
        self.finetune_config = load_config(finetune_config_path)
        self.base_config     = load_config(base_config_path)


        ## Tokenizer
        self.tokenizer           = None
        ## Model
        self.model               = None


        self.MODEL_NAME = model_name
        self.OUT_DIR    = get_root_dir() /  self.finetune_config["model"]["save_path"]
        self.MAX_LEN    = 64            # since it is small dataset go for shorter sequence length -> faster
        self.NUM_EPOCHS = 10            # reduced for speed; increase if needed
        self.BATCH_SIZE = 2
        self.GRAD_ACC   = 1
        self.LR         = 5e-4          # LoRA can use slightly higher LR
        self.LORA_R     = 8
        self.LORA_ALPHA = 32
        self.LORA_DROPOUT = 0.1        



    def load_model(self):
        """Load fine-tuned model and tokenizer for inference."""
        tokenizer = AutoTokenizer.from_pretrained(self.OUT_DIR)
        model = AutoModelForCausalLM.from_pretrained(self.OUT_DIR)
        
        # Enable caching for faster inference
        model.config.use_cache = True
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, self.OUT_DIR)
        model.eval()
        return model, tokenizer

    def run(self, qa_instructions_sft=None, batch_size=4, epochs=3):
        """
        Run supervised instruction fine-tuning on Q&A dataset.
        """
        dataset = Dataset.from_list(qa_instructions_sft)
        self._train(dataset, batch_size=batch_size, epochs=epochs)


    def _train(self, dataset: Dataset, batch_size=8, epochs=3):

        self._init_model_and_tokenizer()

        tokenized_dataset = self._get_tokenized_dataset(dataset)

        self._train_model(tokenized_dataset)



    def _init_model_and_tokenizer(self):
        device          = self.finetune_config["model"]["device"]
        device          = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ## Tokenizer
        self.tokenizer           = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 does not have a pad token

        base_model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME)
        base_model.config.use_cache = False   # ensure loss returned during training
        # wrap with LoRA adapters
        lora_config = LoraConfig(
            r=self.LORA_R, lora_alpha=self.LORA_ALPHA, target_modules=["c_attn", "c_proj"],
            lora_dropout=self.LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        self.model.to(device)

        
    def _train_model(self, tokenized_dataset: Dataset):
        training_args = TrainingArguments(
            output_dir=self.OUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=self.NUM_EPOCHS,
            per_device_train_batch_size=self.BATCH_SIZE,
            gradient_accumulation_steps=self.GRAD_ACC,
            learning_rate=self.LR,
            warmup_steps=5,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="epoch",
            save_total_limit=3,
            fp16=False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=None
        )

        self._inspect_training_plan(trainer)
        
        trainer.train(resume_from_checkpoint=False)
        
        self._save_model()



    def _get_tokenized_dataset(self, dataset: Dataset):
        def tokenize_fn(batch):
            encodings = self.tokenizer(
                batch["text"],
                truncation=True,         #cuts sequences longer than 128 tokens
                padding="max_length",    #pads all sequences to exactly 128 tokens
                max_length=self.MAX_LEN
            )
            #GPT models expect labels same as input_ids for language modeling
            #for causal LM, model uses labels to compute cross-entropy loss
            encodings["labels"] = encodings["input_ids"].copy()  # For causal LM
            return encodings

        #remove_columns=["text"] removes the raw text , keeps only: input_ids (tensor),  attention_mask (tensor), labels (tensor)
        tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        return tokenized_dataset


    def _save_model(self):
        """Save fine-tuned model & tokenizer."""
        self.model.save_pretrained(self.OUT_DIR)
        self.tokenizer.save_pretrained(self.OUT_DIR)
        
        self.logger.info(f"pretrained Model and Tokenizer saved")


    def _inspect_training_plan(self, trainer):
        args = trainer.args
        world_size = args.world_size  # = 1 on single CPU
        num_samples = len(trainer.train_dataset)
        per_device_bsz = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps

        # How many batches the DataLoader yields per epoch
        batches_per_epoch = math.ceil(num_samples / (per_device_bsz * world_size))

        # How many optimizer updates per epoch (this is what logs as "steps")
        update_steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)

        total_update_steps = int(args.num_train_epochs * update_steps_per_epoch)

        print("=== Training Plan ===")
        print(f"num_train_samples           : {num_samples}")
        print(f"world_size                  : {world_size}")
        print(f"per_device_train_batch_size : {per_device_bsz}")
        print(f"gradient_accumulation_steps : {grad_accum}")
        print(f"batches_per_epoch           : {batches_per_epoch}")
        print(f"update_steps_per_epoch      : {update_steps_per_epoch}")
        print(f"num_train_epochs            : {args.num_train_epochs}")
        print(f"EXPECTED total update steps : {total_update_steps}")
        print("======================")


    

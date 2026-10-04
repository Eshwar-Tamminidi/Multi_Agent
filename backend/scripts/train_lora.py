"""Train a lightweight LoRA adapter for Qwen 2.5 / Llama on enterprise QA pairs.
Usage:
    python scripts/train_lora.py [--steps 60] [--model Qwen/Qwen2.5-1.5B-Instruct]
"""
import argparse
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from app import config
from app.datasets import RAG_QA
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model", type=str, default=config.LLM_MODEL)
    args = parser.parse_args()

    print(f"Loading base model {args.model} for LoRA fine-tuning...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if config.DEVICE in ("cuda", "mps") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Prepare enterprise dataset
    records = []
    for q, kws, src in RAG_QA:
        prompt = f"<|im_start|>system\nYou are Nimbus Assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\nAccording to enterprise policy ({src}), " + " ".join(kws) + ".<|im_end|>"
        records.append({"text": prompt})

    ds = Dataset.from_list(records)
    def tokenize(ex):
        res = tok(ex["text"], truncation=True, max_length=128, padding="max_length")
        res["labels"] = res["input_ids"].copy()
        return res

    ds = ds.map(tokenize, batched=True)

    out_dir = config.LORA_ADAPTER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "tmp"),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=(config.DEVICE == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
    )
    print("Starting LoRA fine-tuning...")
    trainer.train()
    print(f"Saving LoRA adapter to {out_dir}...")
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print("LoRA adapter saved successfully!")

if __name__ == "__main__":
    main()

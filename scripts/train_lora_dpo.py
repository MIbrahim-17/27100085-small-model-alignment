import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
parser.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
parser.add_argument("--output_dir", default="outputs/lora_dpo")
parser.add_argument("--val_size", type=int, default=20)
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--grad_accum", type=int, default=2)
parser.add_argument("--epochs", type=float, default=3)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with open(args.data_path) as f:
    rows = json.load(f)

records = []
for row in rows:
    records.append(
        {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }
    )

dataset = Dataset.from_list(records)
split = dataset.train_test_split(test_size=args.val_size, seed=args.seed)
train_dataset = split["train"]
val_dataset = split["test"]
print("train pairs:", len(train_dataset))
print("validation pairs:", len(val_dataset))

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

peft_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

training_args = DPOConfig(
    output_dir=args.output_dir,
    beta=args.beta,
    max_length=args.max_length,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available(),
    bf16=False,
    gradient_checkpointing=False,
    seed=args.seed,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

adapter_dir = os.path.join(args.output_dir, "final")
trainer.save_model(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

with open(os.path.join(args.output_dir, "metrics.jsonl"), "w") as f:
    for record in trainer.state.log_history:
        f.write(json.dumps(record) + "\n")

with open(os.path.join(args.output_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

print("saved LoRA adapter to", adapter_dir)

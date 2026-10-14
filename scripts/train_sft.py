import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from alignment_core import build_sft_batch, compute_batch_logprobs, compute_sft_loss, count_response_tokens, format_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--output_dir", default="outputs/sft")
parser.add_argument("--train_size", type=int, default=30000)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--grad_accum", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--warmup_fraction", type=float, default=0.03)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--eval_every", type=int, default=100)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

os.makedirs(args.output_dir, exist_ok=True)
metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
best_dir = os.path.join(args.output_dir, "best")
final_dir = os.path.join(args.output_dir, "final")

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
use_amp = device != "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
model.to(device)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
dataset = dataset.shuffle(seed=args.seed)

examples = []
for row in dataset:
    messages = row["messages"]
    if len(messages) < 2:
        continue
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        continue
    instruction = messages[0]["content"].strip()
    response = messages[1]["content"].strip()
    if instruction == "" or response == "":
        continue
    examples.append({"prompt": format_prompt(instruction), "response": response})
    if len(examples) == args.train_size + args.val_size:
        break

train_examples = examples[: args.train_size]
val_examples = examples[args.train_size :]
print("train examples:", len(train_examples))
print("validation examples:", len(val_examples))

with open(os.path.join(args.output_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)


def write_metrics(record):
    with open(metrics_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_validation():
    model.eval()
    total_negative_logprob = 0.0
    total_tokens = 0
    val_batch_size = 2
    for start in range(0, len(val_examples), val_batch_size):
        rows = val_examples[start : start + val_batch_size]
        prompts = [row["prompt"] for row in rows]
        responses = [row["response"] for row in rows]
        batch = build_sft_batch(tokenizer, prompts, responses, args.max_length)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logprobs = compute_batch_logprobs(model, batch)
        total_negative_logprob += -logprobs.sum().item()
        total_tokens += count_response_tokens(batch).item()
    model.train()
    return total_negative_logprob / max(total_tokens, 1)


optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

batches_per_epoch = len(train_examples) // args.batch_size
total_steps = (batches_per_epoch * args.epochs) // args.grad_accum
warmup_steps = int(args.warmup_fraction * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
print("optimizer steps:", total_steps)


def validate_and_save(best_val_loss):
    val_loss = run_validation()
    print("step", optimizer_step, "validation loss", val_loss)
    write_metrics({"step": optimizer_step, "val_loss": val_loss})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print("saved best checkpoint at step", optimizer_step)
    return best_val_loss


best_val_loss = float("inf")
last_eval_step = 0
optimizer_step = 0
micro_step = 0
running_loss = 0.0
running_count = 0

val_loss = run_validation()
print("step 0 validation loss", val_loss)
write_metrics({"step": 0, "val_loss": val_loss})

model.train()
optimizer.zero_grad()

for epoch in range(args.epochs):
    random.shuffle(train_examples)
    for start in range(0, batches_per_epoch * args.batch_size, args.batch_size):
        rows = train_examples[start : start + args.batch_size]
        prompts = [row["prompt"] for row in rows]
        responses = [row["response"] for row in rows]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = compute_sft_loss(model, tokenizer, prompts, responses, args.max_length)

        scaler.scale(loss / args.grad_accum).backward()
        running_loss += loss.item()
        running_count += 1
        micro_step += 1

        if micro_step % args.grad_accum != 0:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        optimizer_step += 1

        if optimizer_step % args.log_every == 0:
            train_loss = running_loss / running_count
            running_loss = 0.0
            running_count = 0
            learning_rate = scheduler.get_last_lr()[0]
            print("step", optimizer_step, "train loss", train_loss, "lr", learning_rate)
            write_metrics({"step": optimizer_step, "train_loss": train_loss, "learning_rate": learning_rate})

        if optimizer_step % args.eval_every == 0:
            best_val_loss = validate_and_save(best_val_loss)
            last_eval_step = optimizer_step

if last_eval_step != optimizer_step:
    best_val_loss = validate_and_save(best_val_loss)

model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
    json.dump({"best_val_loss": best_val_loss, "optimizer_steps": optimizer_step}, f, indent=2)
print("done. best validation loss:", best_val_loss)

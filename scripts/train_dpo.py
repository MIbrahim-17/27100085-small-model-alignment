import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment_core import build_dpo_batch, compute_batch_logprobs, compute_dpo_loss, compute_dpo_loss_from_logps, format_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["hh", "ultrafeedback"], required=True)
parser.add_argument("--sft_path", default="outputs/sft/best")
parser.add_argument("--output_dir", default="outputs/dpo")
parser.add_argument("--train_size", type=int, default=15000)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--max_prompt_length", type=int, default=256)
parser.add_argument("--grad_accum", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--beta", type=float, default=0.1)
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
    policy_device = "cuda:0"
    if torch.cuda.device_count() > 1:
        reference_device = "cuda:1"
    else:
        reference_device = "cuda:0"
else:
    policy_device = "cpu"
    reference_device = "cpu"
use_amp = policy_device != "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32)
policy_model.to(policy_device)

reference_model = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32)
reference_model.to(reference_device)
reference_model.eval()
for parameter in reference_model.parameters():
    parameter.requires_grad = False


def load_hh_pairs():
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.shuffle(seed=args.seed)
    human_marker = "\n\nHuman:"
    assistant_marker = "\n\nAssistant:"
    pairs = []
    for row in dataset:
        chosen_text = row["chosen"]
        rejected_text = row["rejected"]
        if chosen_text.count(human_marker) != 1 or chosen_text.count(assistant_marker) != 1:
            continue
        if rejected_text.count(human_marker) != 1 or rejected_text.count(assistant_marker) != 1:
            continue
        chosen_split = chosen_text.index(assistant_marker)
        rejected_split = rejected_text.index(assistant_marker)
        chosen_prompt = chosen_text[:chosen_split]
        rejected_prompt = rejected_text[:rejected_split]
        if chosen_prompt != rejected_prompt:
            continue
        instruction = chosen_prompt.replace(human_marker, "").strip()
        chosen = chosen_text[chosen_split + len(assistant_marker) :].strip()
        rejected = rejected_text[rejected_split + len(assistant_marker) :].strip()
        pairs.append({"instruction": instruction, "chosen": chosen, "rejected": rejected})
    return pairs


def load_ultrafeedback_pairs():
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    dataset = dataset.shuffle(seed=args.seed)
    pairs = []
    for row in dataset:
        if len(row["chosen"]) != 2 or len(row["rejected"]) != 2:
            continue
        instruction = row["prompt"].strip()
        chosen = row["chosen"][1]["content"].strip()
        rejected = row["rejected"][1]["content"].strip()
        pairs.append({"instruction": instruction, "chosen": chosen, "rejected": rejected})
    return pairs


if args.dataset == "hh":
    raw_pairs = load_hh_pairs()
else:
    raw_pairs = load_ultrafeedback_pairs()
print("single-turn pairs found:", len(raw_pairs))

pairs = []
for pair in raw_pairs:
    if pair["instruction"] == "" or pair["chosen"] == "" or pair["rejected"] == "":
        continue
    if pair["chosen"] == pair["rejected"]:
        continue
    prompt = format_prompt(pair["instruction"])
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_length > args.max_prompt_length:
        continue
    pairs.append({"prompt": prompt, "chosen": pair["chosen"], "rejected": pair["rejected"]})
    if len(pairs) == args.train_size + args.val_size:
        break

train_pairs = pairs[: args.train_size]
val_pairs = pairs[args.train_size :]
print("train pairs:", len(train_pairs))
print("validation pairs:", len(val_pairs))

with open(os.path.join(args.output_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)


def write_metrics(record):
    with open(metrics_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def compute_val_logps(model):
    chosen_logps = []
    rejected_logps = []
    val_batch_size = 2
    for start in range(0, len(val_pairs), val_batch_size):
        rows = val_pairs[start : start + val_batch_size]
        prompts = [row["prompt"] for row in rows]
        chosen = [row["chosen"] for row in rows]
        rejected = [row["rejected"] for row in rows]
        batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, args.max_length)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                chosen_logps.append(compute_batch_logprobs(model, batch["chosen"]).float().cpu())
                rejected_logps.append(compute_batch_logprobs(model, batch["rejected"]).float().cpu())
    return torch.cat(chosen_logps), torch.cat(rejected_logps)


reference_val_chosen, reference_val_rejected = compute_val_logps(reference_model)


def run_validation():
    policy_model.eval()
    policy_val_chosen, policy_val_rejected = compute_val_logps(policy_model)
    policy_model.train()
    val_loss = compute_dpo_loss_from_logps(
        policy_val_chosen,
        policy_val_rejected,
        reference_val_chosen,
        reference_val_rejected,
        args.beta,
    ).item()
    chosen_rewards = args.beta * (policy_val_chosen - reference_val_chosen)
    rejected_rewards = args.beta * (policy_val_rejected - reference_val_rejected)
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
    reward_margin = (chosen_rewards - rejected_rewards).mean().item()
    return val_loss, reward_accuracy, reward_margin


def validate_and_save(best_accuracy):
    val_loss, reward_accuracy, reward_margin = run_validation()
    print("step", optimizer_step, "val loss", val_loss, "reward accuracy", reward_accuracy, "reward margin", reward_margin)
    write_metrics(
        {
            "step": optimizer_step,
            "val_loss": val_loss,
            "reward_accuracy": reward_accuracy,
            "reward_margin": reward_margin,
        }
    )
    if optimizer_step > 0 and reward_accuracy > best_accuracy:
        best_accuracy = reward_accuracy
        policy_model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print("saved best checkpoint at step", optimizer_step)
    return best_accuracy


optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

best_accuracy = -1.0
last_eval_step = 0
optimizer_step = 0
micro_step = 0
running_loss = 0.0
running_count = 0

best_accuracy = validate_and_save(best_accuracy)

policy_model.train()
optimizer.zero_grad()

for epoch in range(args.epochs):
    random.shuffle(train_pairs)
    for pair in train_pairs:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = compute_dpo_loss(
                policy_model,
                reference_model,
                tokenizer,
                [pair["prompt"]],
                [pair["chosen"]],
                [pair["rejected"]],
                args.beta,
                args.max_length,
            )

        scaler.scale(loss / args.grad_accum).backward()
        running_loss += loss.item()
        running_count += 1
        micro_step += 1

        if micro_step % args.grad_accum != 0:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        optimizer_step += 1

        if optimizer_step % args.log_every == 0:
            train_loss = running_loss / running_count
            running_loss = 0.0
            running_count = 0
            print("step", optimizer_step, "train loss", train_loss)
            write_metrics({"step": optimizer_step, "train_loss": train_loss})

        if optimizer_step % args.eval_every == 0:
            best_accuracy = validate_and_save(best_accuracy)
            last_eval_step = optimizer_step

if last_eval_step != optimizer_step:
    best_accuracy = validate_and_save(best_accuracy)

policy_model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
    json.dump({"best_reward_accuracy": best_accuracy, "optimizer_steps": optimizer_step}, f, indent=2)
print("done. best validation reward accuracy:", best_accuracy)

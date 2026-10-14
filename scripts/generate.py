import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment_core import format_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--adapter_path", default="")
parser.add_argument("--eval_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--generator", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--prompt_format", choices=["plain", "chat"], default="plain")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda:0"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
if args.adapter_path != "":
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter_path)
model.to(device)
model.eval()

with open(args.eval_path) as f:
    eval_rows = json.load(f)
if args.limit > 0:
    eval_rows = eval_rows[: args.limit]

results = []
if os.path.exists(args.output_path):
    with open(args.output_path) as f:
        results = json.load(f)
done_ids = set()
for result in results:
    done_ids.add(result["id"])

todo_rows = []
for row in eval_rows:
    if row["id"] not in done_ids:
        todo_rows.append(row)
print("prompts to generate:", len(todo_rows))

output_folder = os.path.dirname(args.output_path)
if output_folder != "":
    os.makedirs(output_folder, exist_ok=True)

for start in range(0, len(todo_rows), args.batch_size):
    rows = todo_rows[start : start + args.batch_size]

    prompts = []
    for row in rows:
        if args.prompt_format == "chat":
            messages = [{"role": "user", "content": row["instruction"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = format_prompt(row["instruction"])
        prompts.append(prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    for i in range(len(rows)):
        new_ids = output_ids[i][input_ids.shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        result = {
            "id": rows[i]["id"],
            "instruction": rows[i]["instruction"],
            "output": text,
            "generator": args.generator,
            "dataset": args.dataset,
        }
        if "category" in rows[i]:
            result["category"] = rows[i]["category"]
        if "expected_behavior" in rows[i]:
            result["expected_behavior"] = rows[i]["expected_behavior"]
        results.append(result)

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("generated", len(results), "of", len(eval_rows))

for result in results[:3]:
    print("INSTRUCTION:", result["instruction"])
    print("OUTPUT:", result["output"])
    print("-----")

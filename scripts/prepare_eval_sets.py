import argparse
import json
import os
import random

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default="eval_sets/alpaca_eval_300_seed42.json")
parser.add_argument("--num_prompts", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

file_path = hf_hub_download(repo_id="tatsu-lab/alpaca_eval", filename="alpaca_eval.json", repo_type="dataset")
with open(file_path) as f:
    all_rows = json.load(f)
print("alpaca eval rows:", len(all_rows))

random.seed(args.seed)
indices = random.sample(range(len(all_rows)), args.num_prompts)
indices.sort()

eval_rows = []
for index in indices:
    eval_rows.append(
        {
            "id": "alpaca-eval-" + str(args.seed) + "-" + str(index).zfill(4),
            "instruction": all_rows[index]["instruction"],
            "dataset": "alpaca_eval",
        }
    )

output_folder = os.path.dirname(args.output_path)
if output_folder != "":
    os.makedirs(output_folder, exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(eval_rows, f, indent=2)
print("wrote", len(eval_rows), "prompts to", args.output_path)

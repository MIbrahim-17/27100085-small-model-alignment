import argparse
import json
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--baseline_path", required=True)
parser.add_argument("--candidate_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

with open(args.baseline_path) as f:
    baseline_rows = json.load(f)
with open(args.candidate_path) as f:
    candidate_rows = json.load(f)

candidate_by_id = {}
for row in candidate_rows:
    candidate_by_id[row["id"]] = row

output_folder = os.path.dirname(args.output_path)
if output_folder != "":
    os.makedirs(output_folder, exist_ok=True)

count = 0
with open(args.output_path, "w") as f:
    for baseline_row in baseline_rows:
        if baseline_row["id"] not in candidate_by_id:
            continue
        candidate_row = candidate_by_id[baseline_row["id"]]

        candidate_is_a = random.random() < 0.5
        if candidate_is_a:
            row_a = candidate_row
            row_b = baseline_row
        else:
            row_a = baseline_row
            row_b = candidate_row

        pair = {
            "id": baseline_row["id"],
            "instruction": baseline_row["instruction"],
            "response_a": row_a["output"],
            "response_b": row_b["output"],
            "model_a": row_a["generator"],
            "model_b": row_b["generator"],
            "candidate": candidate_row["generator"],
            "baseline": baseline_row["generator"],
        }
        if "category" in baseline_row:
            pair["category"] = baseline_row["category"]
        if "expected_behavior" in baseline_row:
            pair["expected_behavior"] = baseline_row["expected_behavior"]

        f.write(json.dumps(pair) + "\n")
        count += 1

print("wrote", count, "pairs to", args.output_path)

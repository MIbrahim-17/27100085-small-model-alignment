import argparse
import json

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default="outputs/eval/ultrafeedback_length_stats.json")
args = parser.parse_args()

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

pairs = 0
chosen_words = 0
rejected_words = 0
chosen_longer = 0
for row in dataset:
    if len(row["chosen"]) != 2 or len(row["rejected"]) != 2:
        continue
    chosen_count = len(row["chosen"][1]["content"].split())
    rejected_count = len(row["rejected"][1]["content"].split())
    pairs += 1
    chosen_words += chosen_count
    rejected_words += rejected_count
    if chosen_count > rejected_count:
        chosen_longer += 1

results = {
    "pairs": pairs,
    "chosen_mean_words": chosen_words / pairs,
    "rejected_mean_words": rejected_words / pairs,
    "chosen_longer_fraction": chosen_longer / pairs,
}

with open(args.output_path, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

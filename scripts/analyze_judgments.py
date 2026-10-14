import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--pairs_path", required=True)
parser.add_argument("--judged_path", required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

REFUSAL_PHRASES = [
    "i cannot",
    "i can't",
    "i can not",
    "i'm sorry",
    "i am sorry",
    "i am not able",
    "i'm not able",
    "i do not have the capability",
    "not appropriate",
    "i am not allowed",
    "cannot provide",
    "can't provide",
    "i won't",
    "i will not",
]

REASON_PATTERNS = {
    "hallucination": "hallucinat",
    "cut_off": "truncat|cuts? off",
    "refusal": "refus",
}


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip() != "":
                rows.append(json.loads(line))
    return rows


def has_refusal_phrase(text):
    start = text[:200].lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in start:
            return True
    return False


def ends_with_punctuation(text):
    text = text.strip()
    if text == "":
        return False
    return text[-1] in ".!?\"')]"


pairs = {}
for pair in read_jsonl(args.pairs_path):
    pairs[pair["id"]] = pair
judgments = read_jsonl(args.judged_path)

results = {
    "pairs": len(judgments),
    "wins": 0,
    "losses": 0,
    "ties": 0,
    "picked_response_a": 0,
    "picked_response_b": 0,
    "candidate_shown_as_a": 0,
    "candidate_word_total": 0,
    "baseline_word_total": 0,
    "candidate_outputs_170_words_or_more": 0,
    "baseline_outputs_170_words_or_more": 0,
    "candidate_outputs_without_final_punctuation": 0,
    "baseline_outputs_without_final_punctuation": 0,
    "candidate_refusal_phrase_outputs": 0,
    "baseline_refusal_phrase_outputs": 0,
    "length_buckets": {
        "candidate_longer_by_20_percent": {"wins": 0, "losses": 0, "ties": 0},
        "similar_length": {"wins": 0, "losses": 0, "ties": 0},
        "candidate_shorter_by_20_percent": {"wins": 0, "losses": 0, "ties": 0},
    },
    "reason_mentions": {},
    "categories": {},
}
for name in REASON_PATTERNS:
    results["reason_mentions"][name] = {"wins": 0, "losses": 0, "ties": 0}

for judgment in judgments:
    pair = pairs[judgment["id"]]
    if pair["model_a"] == judgment["candidate"]:
        candidate_text = pair["response_a"]
        baseline_text = pair["response_b"]
        results["candidate_shown_as_a"] += 1
    else:
        candidate_text = pair["response_b"]
        baseline_text = pair["response_a"]

    if judgment["winner_model"] == judgment["candidate"]:
        outcome = "wins"
    elif judgment["winner_model"] == judgment["baseline"]:
        outcome = "losses"
    else:
        outcome = "ties"
    results[outcome] += 1

    if judgment["winner_position"] == "A":
        results["picked_response_a"] += 1
    if judgment["winner_position"] == "B":
        results["picked_response_b"] += 1

    candidate_words = len(candidate_text.split())
    baseline_words = len(baseline_text.split())
    results["candidate_word_total"] += candidate_words
    results["baseline_word_total"] += baseline_words
    if candidate_words >= 170:
        results["candidate_outputs_170_words_or_more"] += 1
    if baseline_words >= 170:
        results["baseline_outputs_170_words_or_more"] += 1
    if not ends_with_punctuation(candidate_text):
        results["candidate_outputs_without_final_punctuation"] += 1
    if not ends_with_punctuation(baseline_text):
        results["baseline_outputs_without_final_punctuation"] += 1
    if has_refusal_phrase(candidate_text):
        results["candidate_refusal_phrase_outputs"] += 1
    if has_refusal_phrase(baseline_text):
        results["baseline_refusal_phrase_outputs"] += 1

    if candidate_words > 1.2 * baseline_words:
        bucket = "candidate_longer_by_20_percent"
    elif candidate_words < 0.8 * baseline_words:
        bucket = "candidate_shorter_by_20_percent"
    else:
        bucket = "similar_length"
    results["length_buckets"][bucket][outcome] += 1

    for name in REASON_PATTERNS:
        if re.search(REASON_PATTERNS[name], judgment["reason"], re.IGNORECASE):
            results["reason_mentions"][name][outcome] += 1

    category = judgment.get("category", "")
    if category != "":
        if category not in results["categories"]:
            results["categories"][category] = {"wins": 0, "losses": 0, "ties": 0}
        results["categories"][category][outcome] += 1

results["candidate_mean_words"] = results["candidate_word_total"] / max(len(judgments), 1)
results["baseline_mean_words"] = results["baseline_word_total"] / max(len(judgments), 1)

with open(args.output_path, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

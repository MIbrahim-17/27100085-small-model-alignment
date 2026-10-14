# Small-Model Alignment Starter

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.

## Solution Overview

This repository contains a solution for the SFT + DPO alignment assignment. It trains `Qwen/Qwen2.5-0.5B` with supervised fine-tuning, runs two from-scratch DPO trainings (HH-RLHF and UltraFeedback), runs a TRL `DPOTrainer` + LoRA persona experiment on `HuggingFaceTB/SmolLM2-360M-Instruct`, and evaluates DPO against SFT with a pairwise model judge.

### Layout

```text
alignment_core.py              batching, response-only log-probs, SFT loss, DPO loss
submission_adapter.py          grading interface, thin wrappers around alignment_core.py
scripts/prepare_eval_sets.py   builds the fixed AlpacaEval subset (300 prompts, seed 42)
scripts/train_sft.py           from-scratch SFT training loop
scripts/train_dpo.py           from-scratch DPO training loop (--dataset hh or ultrafeedback)
scripts/train_lora_dpo.py      TRL DPOTrainer + PEFT LoRA persona run
scripts/generate.py            greedy generation on a fixed evaluation set
scripts/build_pairs.py         randomized A/B pair files for DPO vs SFT
scripts/judge_pairwise_kaggle.py  resumable pairwise judging with kaggle_benchmarks
scripts/plot_metrics.py        training plots from metrics.jsonl
scripts/analyze_judgments.py   length, keyword, and category statistics from judged pairs
scripts/measure_preference_lengths.py  chosen vs rejected response length in UltraFeedback
notebooks/                     Kaggle notebooks that call the scripts in order
eval_sets/                     fixed evaluation prompt sets
data/persona_lora/             persona preference data
tests/                         public adapter tests and extra edge-case tests
outputs/                       metrics, plots, generations, pairs, and judgments
report/                        final report
```

### Prompt format

The same plain-text template is used for SFT, DPO, and generation with the Qwen models:

```text
### Instruction:
{instruction}

### Response:
{response}<|endoftext|>
```

The LoRA persona run uses the SmolLM2 chat template through TRL.

### SFT loss reduction

`compute_sft_loss` returns the mean negative log-probability over all unmasked response tokens in the batch.

### Tests

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch pytest --index-url https://download.pytorch.org/whl/cpu
PYTHONPATH=. .venv/bin/python -m pytest -q tests
```

### Running on Kaggle (GPU T4 x2)

Run the notebooks in this order. Each notebook copies the repository to `/kaggle/working/small-model-alignment`, installs the dependencies without reinstalling PyTorch, and writes everything under `outputs/`.

| Notebook | What it does | Inputs |
|---|---|---|
| `notebooks/01_sft.ipynb` | builds the AlpacaEval subset, base generation, SFT training, SFT generation | code |
| `notebooks/02_dpo_hh.ipynb` | HH-RLHF DPO, generation on the helpful/harmless set, pair file | output of 01 |
| `notebooks/03_dpo_ultrafeedback.ipynb` | UltraFeedback DPO, generation on AlpacaEval, pair file | output of 01 |
| `notebooks/04_lora_persona.ipynb` | LoRA-DPO persona training, generation, pair file | code |
| `notebooks/05_judge.ipynb` | pairwise judging from the saved pair files | outputs of 02, 03, 04 |

`notebooks/00_smoke.ipynb` runs the whole pipeline with tiny sizes to check that everything works.

`notebooks/04b_lora_persona_pilot_5_epochs.ipynb` is the 5-epoch LoRA comparison run reported in the report. `notebooks/build_notebooks.py` writes all the notebooks and their Kaggle metadata, and `notebooks/stage_code_dataset.sh` copies the code into a folder that can be uploaded as a Kaggle dataset.

### Judging

`scripts/judge_pairwise_kaggle.py` needs the Kaggle model proxy. `notebooks/05_judge.ipynb` is meant for a Kaggle benchmark notebook, where the proxy is already set up. The reported judgments were produced outside a notebook, by creating a short-lived token first and then running the script:

```bash
kaggle benchmarks auth -y --env-file judge.env
set -a; . ./judge.env; set +a
export LLM_DEFAULT=google/gemini-3.6-flash LLM_DEFAULT_EVAL=google/gemini-3.6-flash LLMS_AVAILABLE=google/gemini-3.6-flash
python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_hh_vs_sft_pairs.jsonl --output_path outputs/eval/dpo_hh_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash --judge_mode safety --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
```

The judge modes are `general` (UltraFeedback DPO on AlpacaEval), `safety` (HH-RLHF DPO on the helpful/harmless set), and `persona` (LoRA-DPO).

When resuming with `--reuse_judgments`, pairs that already have a valid verdict are skipped and pairs with an invalid verdict are judged again.

The extra LoRA usefulness and safety comparison uses the persona pair file with `--judge_mode safety` and `--output_path outputs/eval/lora_dpo_vs_base_judged_usefulness_safety.jsonl`. The 5-epoch LoRA comparison was judged the same two ways from `outputs/lora_dpo_pilot_5_epochs/lora_dpo_vs_base_pairs.jsonl`.

The statistics quoted in the report (word counts, length buckets, judge-reason keyword counts, position counts, per-category results) come from:

```bash
python scripts/analyze_judgments.py --pairs_path outputs/eval/dpo_hh_vs_sft_pairs.jsonl --judged_path outputs/eval/dpo_hh_vs_sft_judged.jsonl --output_path outputs/eval/dpo_hh_vs_sft_analysis.json
```

### Results

The final report is `report/report.pdf` (source `report/report.md`). Metrics, plots, generations, pair files, and judged outputs are under `outputs/`.

The same steps as commands:

```bash
python scripts/prepare_eval_sets.py
python scripts/train_sft.py --output_dir outputs/sft
python scripts/train_dpo.py --dataset hh --sft_path outputs/sft/best --output_dir outputs/dpo_hh
python scripts/train_dpo.py --dataset ultrafeedback --sft_path outputs/sft/best --output_dir outputs/dpo_ultrafeedback
python scripts/train_lora_dpo.py --output_dir outputs/lora_dpo
```

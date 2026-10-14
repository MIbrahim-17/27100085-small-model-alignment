import json
import os

USER = "mibrahim1717"
CODE_DATASET = USER + "/pa2-code"

SETUP = [
    """import glob
import os
import shutil

REPO = "/kaggle/working/small-model-alignment"
found = glob.glob("/kaggle/input/**/pa2-code/**/alignment_core.py", recursive=True)
print(found)
if not os.path.exists(REPO):
    shutil.copytree(os.path.dirname(found[0]), REPO)
os.chdir(REPO)
os.environ["PYTHONPATH"] = REPO
print(os.listdir(REPO))""",
    """!nvidia-smi
!python -m pip uninstall -y -q torchao || true
!python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
!python -m pip install -q -e . --no-deps""",
    """import datasets
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("gpus", torch.cuda.device_count())
print("model proxy available", "MODEL_PROXY_URL" in os.environ)""",
]

LORA_INSTALL = '!python -m pip install -q "trl==0.29.1" "peft>=0.17,<0.22" "transformers>=4.56.2,<5" "datasets>=3.0,<5" "accelerate>=1.4.0"'

FIND_SFT = """sft_config = glob.glob("/kaggle/input/**/outputs/sft/best/config.json", recursive=True)
print(sft_config)
SFT_PATH = os.path.dirname(sft_config[0])
SFT_GENERATIONS = os.path.dirname(glob.glob("/kaggle/input/**/outputs/generations/sft_alpaca.json", recursive=True)[0])
ALPACA_EVAL = glob.glob("/kaggle/input/**/eval_sets/alpaca_eval_300_seed42.json", recursive=True)[0]
print(SFT_PATH, SFT_GENERATIONS, ALPACA_EVAL)"""


def sft_cells(train_size, val_size, eval_every, limit, gen):
    return [
        "!PYTHONPATH=. python -m pytest -q tests",
        "!python scripts/prepare_eval_sets.py",
        "!python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --eval_path eval_sets/alpaca_eval_300_seed42.json --output_path outputs/generations/sanity_base_alpaca.json --generator qwen2.5-0.5b-base --dataset alpaca_eval --limit 3 " + gen,
        "!python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --eval_path eval_sets/alpaca_eval_300_seed42.json --output_path outputs/generations/base_alpaca.json --generator qwen2.5-0.5b-base --dataset alpaca_eval " + limit + gen,
        "!python scripts/train_sft.py --output_dir outputs/sft --train_size " + str(train_size) + " --val_size " + str(val_size) + " --eval_every " + str(eval_every),
        "!python scripts/plot_metrics.py --metrics_path outputs/sft/metrics.jsonl --output_dir outputs/plots --run_name sft --kind sft",
        "!python scripts/generate.py --model_path outputs/sft/best --eval_path eval_sets/alpaca_eval_300_seed42.json --output_path outputs/generations/sanity_sft_alpaca.json --generator qwen2.5-0.5b-sft --dataset alpaca_eval --limit 3 " + gen,
        "!python scripts/generate.py --model_path outputs/sft/best --eval_path eval_sets/alpaca_eval_300_seed42.json --output_path outputs/generations/sft_alpaca.json --generator qwen2.5-0.5b-sft --dataset alpaca_eval " + limit + gen,
        "!python scripts/generate.py --model_path outputs/sft/best --eval_path eval_sets/hh_rlhf_eval_300_seed42.json --output_path outputs/generations/sft_hh.json --generator qwen2.5-0.5b-sft --dataset hh_rlhf_style_eval " + limit + gen,
        "!zip -q -r /kaggle/working/sft_run_outputs.zip outputs eval_sets -x 'outputs/*/best/*' 'outputs/*/final/*'",
    ]


def dpo_cells(dataset, run, sft_path, sft_generations, alpaca_eval, train_size, val_size, eval_every, limit, gen):
    if dataset == "hh":
        eval_path = "eval_sets/hh_rlhf_eval_300_seed42.json"
        eval_name = "hh_rlhf_style_eval"
        sft_file = "sft_hh.json"
        out_file = "dpo_hh_hh.json"
    else:
        eval_path = alpaca_eval
        eval_name = "alpaca_eval"
        sft_file = "sft_alpaca.json"
        out_file = "dpo_uf_alpaca.json"
    generator = "qwen2.5-0.5b-" + run.replace("_", "-")
    return [
        "!python scripts/train_dpo.py --dataset " + dataset + " --sft_path " + sft_path + " --output_dir outputs/" + run + " --train_size " + str(train_size) + " --val_size " + str(val_size) + " --eval_every " + str(eval_every),
        "!python scripts/plot_metrics.py --metrics_path outputs/" + run + "/metrics.jsonl --output_dir outputs/plots --run_name " + run + " --kind dpo",
        "!python scripts/generate.py --model_path outputs/" + run + "/best --eval_path " + eval_path + " --output_path outputs/generations/sanity_" + out_file + " --generator " + generator + " --dataset " + eval_name + " --limit 3 " + gen,
        "!python scripts/generate.py --model_path outputs/" + run + "/best --eval_path " + eval_path + " --output_path outputs/generations/" + out_file + " --generator " + generator + " --dataset " + eval_name + " " + limit + gen,
        "!python scripts/build_pairs.py --baseline_path " + sft_generations + "/" + sft_file + " --candidate_path outputs/generations/" + out_file + " --output_path outputs/eval/" + run + "_vs_sft_pairs.jsonl",
        "!zip -q -r /kaggle/working/" + run + "_run_outputs.zip outputs -x 'outputs/*/best/*' 'outputs/*/final/*'",
    ]


def lora_cells(epochs, gen):
    smol = "HuggingFaceTB/SmolLM2-360M-Instruct"
    common = " --eval_path eval_sets/persona_style_eval.json --dataset persona_style_eval --prompt_format chat " + gen
    return [
        LORA_INSTALL,
        "!python scripts/train_lora_dpo.py --output_dir outputs/lora_dpo --epochs " + str(epochs),
        "!python scripts/plot_metrics.py --metrics_path outputs/lora_dpo/metrics.jsonl --output_dir outputs/plots --run_name lora_dpo --kind lora",
        "!python scripts/generate.py --model_path " + smol + " --adapter_path outputs/lora_dpo/final --output_path outputs/generations/sanity_smol_lora_persona.json --generator smollm2-360m-lora-dpo --limit 3" + common,
        "!python scripts/generate.py --model_path " + smol + " --output_path outputs/generations/smol_base_persona.json --generator smollm2-360m-instruct" + common,
        "!python scripts/generate.py --model_path " + smol + " --adapter_path outputs/lora_dpo/final --output_path outputs/generations/smol_lora_persona.json --generator smollm2-360m-lora-dpo" + common,
        "!python scripts/build_pairs.py --baseline_path outputs/generations/smol_base_persona.json --candidate_path outputs/generations/smol_lora_persona.json --output_path outputs/eval/lora_dpo_vs_base_pairs.jsonl",
        "!zip -q -r /kaggle/working/lora_dpo_run_outputs.zip outputs -x 'outputs/sft/*' 'outputs/dpo_hh/*' 'outputs/dpo_ultrafeedback/*'",
    ]


JUDGE = [
    """import glob
import os
import shutil

REPO = "/kaggle/working/small-model-alignment"
found = glob.glob("/kaggle/input/**/pa2-code/**/alignment_core.py", recursive=True)
if not os.path.exists(REPO):
    shutil.copytree(os.path.dirname(found[0]), REPO)
os.chdir(REPO)
os.makedirs("outputs/eval", exist_ok=True)
for path in glob.glob("/kaggle/input/**/outputs/eval/*_pairs.jsonl", recursive=True):
    shutil.copy(path, "outputs/eval")
for path in glob.glob("/kaggle/input/**/outputs/eval/*_judged.jsonl", recursive=True):
    shutil.copy(path, "outputs/eval")
print(os.listdir("outputs/eval"))""",
    "!python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_ultrafeedback_vs_sft_pairs.jsonl --output_path outputs/eval/dpo_ultrafeedback_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash --judge_mode general --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments",
    "!python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_hh_vs_sft_pairs.jsonl --output_path outputs/eval/dpo_hh_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash --judge_mode safety --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments",
    "!python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/lora_dpo_vs_base_pairs.jsonl --output_path outputs/eval/lora_dpo_vs_base_judged.jsonl --judge_model google/gemini-3.6-flash --judge_mode persona --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments",
]


def write_notebook(file_name, cells):
    notebook_cells = []
    for source in cells:
        if source.startswith("!"):
            source = source + "\nassert _exit_code == 0"
        notebook_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})
    notebook = {
        "cells": notebook_cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(file_name, "w") as f:
        json.dump(notebook, f, indent=1)


def write_kernel(slug, title, notebook_name, cells, kernel_sources, gpu):
    folder = os.path.join("kaggle_build", slug)
    os.makedirs(folder, exist_ok=True)
    write_notebook(os.path.join("notebooks", notebook_name), cells)
    write_notebook(os.path.join(folder, notebook_name), cells)
    metadata = {
        "id": USER + "/" + slug,
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": gpu,
        "enable_internet": "true",
        "dataset_sources": [CODE_DATASET],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    if gpu == "true":
        metadata["machine_shape"] = "NvidiaTeslaT4"
    with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


GEN = "--max_new_tokens 256 --repetition_penalty 1.1"
SMOKE_GEN = "--max_new_tokens 48 --repetition_penalty 1.1"

smoke = SETUP + sft_cells(64, 16, 1, "--limit 4 ", SMOKE_GEN)
smoke = smoke + dpo_cells("hh", "dpo_hh", "outputs/sft/best", "outputs/generations", "eval_sets/alpaca_eval_300_seed42.json", 64, 16, 1, "--limit 4 ", SMOKE_GEN)
smoke = smoke + dpo_cells("ultrafeedback", "dpo_ultrafeedback", "outputs/sft/best", "outputs/generations", "eval_sets/alpaca_eval_300_seed42.json", 64, 16, 1, "--limit 4 ", SMOKE_GEN)
smoke = smoke + lora_cells(1, SMOKE_GEN + " --limit 4")
smoke = smoke + ["!rm -rf outputs/sft/best outputs/sft/final outputs/dpo_hh/best outputs/dpo_hh/final outputs/dpo_ultrafeedback/best outputs/dpo_ultrafeedback/final"]
write_kernel("pa2-smoke", "pa2 smoke", "00_smoke.ipynb", smoke, [], "true")

write_kernel("pa2-sft", "pa2 sft", "01_sft.ipynb", SETUP + sft_cells(30000, 1000, 100, "", GEN), [], "true")

for dataset, run, notebook_name in [("hh", "dpo_hh", "02_dpo_hh.ipynb"), ("ultrafeedback", "dpo_ultrafeedback", "03_dpo_ultrafeedback.ipynb")]:
    cells = SETUP + [FIND_SFT] + dpo_cells(dataset, run, "$SFT_PATH", "$SFT_GENERATIONS", "$ALPACA_EVAL", 15000, 1000, 100, "", GEN)
    slug = "pa2-" + run.replace("_", "-")
    write_kernel(slug, slug.replace("-", " "), notebook_name, cells, [USER + "/pa2-sft"], "true")

write_kernel("pa2-lora-persona", "pa2 lora persona", "04_lora_persona.ipynb", SETUP + lora_cells(3, GEN), [], "true")
write_kernel("pa2-lora-pilot-5-epochs", "pa2 lora pilot 5 epochs", "04b_lora_persona_pilot_5_epochs.ipynb", SETUP + lora_cells(5, GEN), [], "true")

write_kernel(
    "pa2-judge",
    "pa2 judge",
    "05_judge.ipynb",
    JUDGE,
    [USER + "/pa2-dpo-hh", USER + "/pa2-dpo-ultrafeedback", USER + "/pa2-lora-persona"],
    "false",
)
print("notebooks written")

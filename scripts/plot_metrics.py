import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--metrics_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--kind", choices=["sft", "dpo", "lora"], required=True)
args = parser.parse_args()

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

records = []
with open(args.metrics_path) as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))

os.makedirs(args.output_dir, exist_ok=True)


def get_series(key):
    steps = []
    values = []
    for record in records:
        if key in record:
            steps.append(record["step"])
            values.append(record[key])
    return steps, values


def style_axes(axes, title, y_label):
    axes.set_title(title)
    axes.set_xlabel("Optimizer step")
    axes.set_ylabel(y_label)
    axes.grid(True, color="#dddddd", linewidth=0.8)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


def mark_best(axes, steps, values, best_index, label):
    axes.scatter([steps[best_index]], [values[best_index]], s=90, color="#0b0b0b", zorder=3, label=label)


if args.kind == "lora":
    train_steps, train_losses = get_series("loss")
    val_steps, val_losses = get_series("eval_loss")
else:
    train_steps, train_losses = get_series("train_loss")
    val_steps, val_losses = get_series("val_loss")

figure, axes = plt.subplots(figsize=(7, 3.6))
axes.plot(train_steps, train_losses, color=BLUE, linewidth=2, label="Train loss (mean per logging window, no smoothing)")
axes.plot(val_steps, val_losses, color=ORANGE, linewidth=2, marker="o", label="Validation loss")
if args.kind == "sft" and len(val_losses) > 0:
    best_index = val_losses.index(min(val_losses))
    mark_best(axes, val_steps, val_losses, best_index, "Best checkpoint (lowest validation loss)")
    style_axes(axes, args.run_name + ": SFT loss", "Loss (mean negative log-prob per response token)")
else:
    style_axes(axes, args.run_name + ": DPO loss", "DPO loss")
axes.legend(loc="best", fontsize=9, framealpha=0.85)
figure.tight_layout()
figure.savefig(os.path.join(args.output_dir, args.run_name + "_loss.png"), dpi=150)
plt.close(figure)

if args.kind == "dpo":
    accuracy_steps, accuracies = get_series("reward_accuracy")
    margin_steps, margins = get_series("reward_margin")
if args.kind == "lora":
    accuracy_steps, accuracies = get_series("eval_rewards/accuracies")
    margin_steps, margins = get_series("eval_rewards/margins")

if args.kind == "dpo" or args.kind == "lora":
    figure, axes = plt.subplots(figsize=(7, 3.6))
    axes.plot(accuracy_steps, accuracies, color=AQUA, linewidth=2, marker="o", label="Validation reward accuracy")
    if args.kind == "dpo" and len(accuracies) > 0:
        best_index = accuracies.index(max(accuracies))
        mark_best(axes, accuracy_steps, accuracies, best_index, "Best checkpoint (highest reward accuracy)")
    style_axes(axes, args.run_name + ": validation reward accuracy", "Reward accuracy")
    axes.set_ylim(0, 1)
    axes.legend(loc="best", fontsize=9, framealpha=0.85)
    figure.tight_layout()
    figure.savefig(os.path.join(args.output_dir, args.run_name + "_reward_accuracy.png"), dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(7, 3.6))
    axes.plot(margin_steps, margins, color=BLUE, linewidth=2, marker="o", label="Validation reward margin")
    style_axes(axes, args.run_name + ": validation reward margin", "Reward margin (mean of r_w - r_l)")
    axes.legend(loc="best", fontsize=9, framealpha=0.85)
    figure.tight_layout()
    figure.savefig(os.path.join(args.output_dir, args.run_name + "_reward_margin.png"), dpi=150)
    plt.close(figure)

print("saved plots to", args.output_dir)

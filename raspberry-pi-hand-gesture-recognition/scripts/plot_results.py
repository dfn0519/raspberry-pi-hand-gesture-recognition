"""Rebuild the README model-comparison chart from the result CSV."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with (ROOT / "results" / "model_comparison.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    models = [row["model"] for row in rows]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = np.array([[float(row[key]) * 100 for key in metrics] for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(models))
    width = 0.19
    colors = ["#2f6fed", "#38a169", "#d69e2e", "#805ad5"]
    for index, (label, color) in enumerate(zip(labels, colors)):
        bars = ax.bar(x + (index - 1.5) * width, values[:, index], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_title("Hand Gesture Model Comparison", fontsize=16, weight="bold", pad=14)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 105)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "model_comparison.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()

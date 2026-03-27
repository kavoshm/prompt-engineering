"""
Generate figures for the Prompt Engineering module.

Produces:
  - docs/images/prompt_patterns.png — Flowchart of 5 prompt patterns
  - docs/images/comparison_chart.png — Zero-shot vs few-shot vs CoT accuracy
  - docs/images/system_prompt_impact.png — System prompt strategy impact on quality
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# --- Theme ---
plt.style.use("dark_background")
BG_COLOR = "#1a1a2e"
COLORS = ["#4f7cac", "#5a9e8f", "#9b6b9e", "#c47e3a", "#b85450"]
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4e"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_prompt_patterns():
    """Flowchart showing the 5 prompt patterns."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(7, 7.5, "Five Prompt Engineering Patterns for Clinical AI",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=TEXT_COLOR)

    # Central node
    cx, cy = 7, 5.5
    central = FancyBboxPatch((cx - 1.5, cy - 0.4), 3, 0.8,
                              boxstyle="round,pad=0.15", facecolor="#2a2a4e",
                              edgecolor=TEXT_COLOR, linewidth=2)
    ax.add_patch(central)
    ax.text(cx, cy, "Clinical\nNote Input", ha="center", va="center",
            fontsize=11, fontweight="bold", color=TEXT_COLOR)

    # Pattern boxes
    patterns = [
        ("Role + Task\n+ Format", "Define persona,\ntask, output schema", COLORS[0]),
        ("Few-Shot\nExamples", "2-3 labeled examples\nin prompt context", COLORS[1]),
        ("Chain-of-\nThought", "Step-by-step\nreasoning trace", COLORS[2]),
        ("Guardrails", "Safety constraints\n& uncertainty flags", COLORS[3]),
        ("Output\nValidation", "Self-check format,\ncodes, ranges", COLORS[4]),
    ]

    positions = [(2, 3), (5, 1.5), (7, 3), (9, 1.5), (12, 3)]

    for i, ((title, desc, color), (px, py)) in enumerate(zip(patterns, positions)):
        box = FancyBboxPatch((px - 1.3, py - 0.6), 2.6, 1.8,
                              boxstyle="round,pad=0.2", facecolor=color,
                              edgecolor=TEXT_COLOR, linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(px, py + 0.35, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(px, py - 0.35, desc, ha="center", va="center",
                fontsize=8, color="#e0e0e0", style="italic")

        # Arrow from central node
        ax.annotate("", xy=(px, py + 0.6 + 0.6), xytext=(cx, cy - 0.4),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                     connectionstyle="arc3,rad=0.1"))

    # Bottom label
    ax.text(7, 0.3, "Each pattern addresses a different aspect of prompt reliability",
            ha="center", va="center", fontsize=10, color="#888888", style="italic")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "prompt_patterns.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'prompt_patterns.png'}")


def fig_comparison_chart():
    """Bar chart comparing zero-shot vs few-shot vs CoT accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    categories = ["Emergency\nDetection", "Urgency\nClassification", "Specialty\nRouting",
                   "ICD-10\nCoding", "Overall\nAccuracy"]
    zero_shot = [92, 67, 78, 55, 73]
    few_shot =  [96, 83, 92, 72, 86]
    cot =       [100, 100, 95, 78, 93]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width, zero_shot, width, label="Zero-Shot",
                    color=COLORS[4], alpha=0.85, edgecolor="none")
    bars2 = ax.bar(x, few_shot, width, label="Few-Shot",
                    color=COLORS[0], alpha=0.85, edgecolor="none")
    bars3 = ax.bar(x + width, cot, width, label="Chain-of-Thought",
                    color=COLORS[1], alpha=0.85, edgecolor="none")

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f"{int(height)}%", ha="center", va="bottom",
                    fontsize=8, color=TEXT_COLOR)

    ax.set_ylabel("Accuracy (%)", fontsize=12, color=TEXT_COLOR)
    ax.set_title("Prompt Strategy Impact on Clinical Classification Accuracy",
                  fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(0, 115)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, facecolor="#2a2a4e",
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_chart.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'comparison_chart.png'}")


def fig_system_prompt_impact():
    """Grouped bars showing how system prompt strategies affect output quality."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    strategies = ["Minimal\nGeneric", "Persona-\nBased", "Constraint-\nHeavy",
                   "Template-\nDriven", "Safety-\nFirst"]
    metrics = {
        "Clinical Accuracy":   [62, 88, 85, 79, 82],
        "Output Structure":    [35, 60, 90, 98, 92],
        "Safety Compliance":   [20, 55, 82, 65, 96],
        "Actionability":       [45, 92, 78, 70, 88],
    }

    x = np.arange(len(strategies))
    n_metrics = len(metrics)
    width = 0.18

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name,
                       color=COLORS[i], alpha=0.85, edgecolor="none")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f"{int(height)}", ha="center", va="bottom",
                    fontsize=7, color=TEXT_COLOR)

    ax.set_ylabel("Score (0-100)", fontsize=12, color=TEXT_COLOR)
    ax.set_title("System Prompt Strategy Impact on Output Quality Dimensions",
                  fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(0, 115)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, facecolor="#2a2a4e",
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, ncol=2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "system_prompt_impact.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'system_prompt_impact.png'}")


if __name__ == "__main__":
    print("Generating figures for 01-prompt-engineering...")
    fig_prompt_patterns()
    fig_comparison_chart()
    fig_system_prompt_impact()
    print("Done.")

#!/usr/bin/env python3
"""
Generate all publication figures for CogBench-RAG paper.
Saves to figures/ directory as both PDF and PNG.

Usage:
    python scripts/generate_figures.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.systems.bm25 import BM25System
from src.systems.dense import DenseRetrievalSystem
from src.modules.fan_effect import FanEffectModule
from src.modules.encoding_specificity import EncodingSpecificityModule
from src.modules.interference import InterferenceModule
from src.modules.serial_position import SerialPositionModule
from src.modules.retrieval_induced_forgetting import RetrievalInducedForgettingModule

# Style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "bm25": "#2176AE",
    "dense": "#E85D04",
    "human": "#57CC99",
    "control": "#95A5A6",
    "highlight": "#E63946",
}

OUTDIR = "figures"


def run_all_benchmarks():
    """Run all modules on all systems and return results dict."""
    np.random.seed(42)

    systems = {
        "BM25": BM25System({"k1": 1.5, "b": 0.75}),
        "Dense": DenseRetrievalSystem({
            "name": "dense_minilm",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "top_k": 10,
        }),
    }

    modules = {
        "Fan Effect": FanEffectModule({
            "fan_sizes": [1, 2, 5, 10], "n_entities_per_fan": 20,
            "n_queries_per_entity": 3, "top_k": 10,
        }),
        "Encoding Specificity": EncodingSpecificityModule({"n_queries_per_condition": 200}),
        "Interference": InterferenceModule({}),
        "Serial Position": SerialPositionModule({}),
        "RIF": RetrievalInducedForgettingModule({"n_practice_rounds": 3}),
    }

    results = {}
    for sys_name, system in systems.items():
        print(f"Running {sys_name}...")
        for mod_name, module in modules.items():
            print(f"  {mod_name}...", end=" ", flush=True)
            result = module.run(system)
            results[(sys_name, mod_name)] = result
            print(f"CAS={result.cognitive_alignment_score:.3f}")

    return results


def fig1_overview_schematic():
    """
    Figure 1: Conceptual overview — mapping human memory phenomena to RAG benchmark tasks.
    This is a schematic/diagram figure. We create a structured layout.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Figure 1: CogBench-RAG Framework", fontsize=13, fontweight="bold", pad=20)

    # Left column: Human Memory Phenomena
    ax.text(1.5, 6.5, "Human Memory\nPhenomena", ha="center", va="top",
            fontsize=11, fontweight="bold", color=COLORS["human"])

    phenomena = [
        "Encoding Specificity\n(Tulving & Thomson, 1973)",
        "Fan Effect\n(Anderson, 1974)",
        "Interference\n(Underwood, 1957)",
        "Serial Position\n(Murdock, 1962)",
        "Retrieval-Induced\nForgetting (Anderson+, 1994)",
    ]
    for i, text in enumerate(phenomena):
        y = 5.5 - i * 1.1
        ax.add_patch(plt.Rectangle((0.1, y - 0.35), 2.8, 0.7,
                     facecolor="#E8F5E9", edgecolor=COLORS["human"], linewidth=1.2, zorder=2))
        ax.text(1.5, y, text, ha="center", va="center", fontsize=7.5, zorder=3)

    # Arrow column
    for i in range(5):
        y = 5.5 - i * 1.1
        ax.annotate("", xy=(5.2, y), xytext=(3.1, y),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
        ax.text(4.15, y + 0.2, "operationalized\nas", ha="center", va="bottom",
                fontsize=6.5, color="#666", style="italic")

    # Right column: Benchmark Modules
    ax.text(7.5, 6.5, "CogBench-RAG\nBenchmark Modules", ha="center", va="top",
            fontsize=11, fontweight="bold", color=COLORS["bm25"])

    modules = [
        "Context-match vs.\nmismatch retrieval",
        "Accuracy vs. entity\nassociation count",
        "Phase 1 vs. Phase 2\nfact retrieval",
        "Retrieval accuracy by\nindexing position",
        "Unpracticed item access\nafter category practice",
    ]
    for i, text in enumerate(modules):
        y = 5.5 - i * 1.1
        ax.add_patch(plt.Rectangle((5.4, y - 0.35), 4.2, 0.7,
                     facecolor="#E3F2FD", edgecolor=COLORS["bm25"], linewidth=1.2, zorder=2))
        ax.text(7.5, y, text, ha="center", va="center", fontsize=7.5, zorder=3)

    # Bottom: CAS output
    ax.add_patch(FancyBboxPatch((2.5, -0.1), 5, 0.7, boxstyle="round,pad=0.15",
                 facecolor="#FFF3E0", edgecolor=COLORS["dense"], linewidth=1.5, zorder=2))
    ax.text(5, 0.25, "→ Cognitive Alignment Score (CAS) per system",
            ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)

    fig.savefig(f"{OUTDIR}/fig1_overview.pdf")
    fig.savefig(f"{OUTDIR}/fig1_overview.png")
    plt.close(fig)
    print("  Saved fig1_overview")


def fig2_heatmap(results):
    """
    Figure 2: Per-phenomenon results heatmap (systems × phenomena).
    """
    modules = ["Fan Effect", "Encoding Specificity", "Interference",
               "Serial Position", "RIF"]
    systems = ["BM25", "Dense"]

    # Build matrix
    matrix = np.zeros((len(systems), len(modules)))
    for i, sys_name in enumerate(systems):
        for j, mod_name in enumerate(modules):
            r = results[(sys_name, mod_name)]
            matrix[i, j] = r.cognitive_alignment_score

    fig, ax = plt.subplots(figsize=(7, 2.5))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(["Fan\nEffect", "Encoding\nSpecificity", "Interference",
                         "Serial\nPosition", "Retrieval-Induced\nForgetting"],
                        fontsize=8)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems, fontsize=10)

    # Annotate cells
    for i in range(len(systems)):
        for j in range(len(modules)):
            r = results[(systems[i], modules[j])]
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            direction = r.direction
            symbol = "✓" if direction == "human_like" else "—"
            ax.text(j, i, f"{val:.2f}\n{symbol}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cognitive Alignment Score", fontsize=9)

    ax.set_title("Figure 2: Cognitive Alignment Across Systems and Phenomena",
                 fontsize=11, fontweight="bold", pad=10)

    fig.savefig(f"{OUTDIR}/fig2_heatmap.pdf")
    fig.savefig(f"{OUTDIR}/fig2_heatmap.png")
    plt.close(fig)
    print("  Saved fig2_heatmap")


def fig3_encoding_specificity(results):
    """
    Figure 3: Encoding specificity deep-dive — BM25 vs Dense across conditions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

    conditions = ["context_match", "context_mismatch", "unrelated"]
    labels = ["Context\nMatch", "Context\nMismatch", "Unrelated"]
    x = np.arange(len(conditions))
    width = 0.35

    for ax, sys_name, color, title in [
        (ax1, "BM25", COLORS["bm25"], "BM25 (Lexical)"),
        (ax2, "Dense", COLORS["dense"], "Dense Retrieval (Semantic)")
    ]:
        r = results[(sys_name, "Encoding Specificity")]
        mrrs = [r.conditions[c]["mrr"] for c in conditions]

        bars = ax.bar(x, mrrs, width=0.6, color=color, alpha=0.85, edgecolor="white")

        # Value labels
        for bar, val in zip(bars, mrrs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Mean Reciprocal Rank" if ax == ax1 else "")
        ax.set_title(title, fontsize=10, fontweight="bold")

        # Add effect size annotation
        gap = mrrs[0] - mrrs[1]
        ax.annotate(f"Δ = {gap:.3f}", xy=(0.5, max(mrrs[0], mrrs[1]) + 0.06),
                    fontsize=9, ha="center", color=COLORS["highlight"], fontweight="bold")

    fig.suptitle("Figure 3: Encoding Specificity Effect by Retrieval Architecture",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_encoding_specificity.pdf")
    fig.savefig(f"{OUTDIR}/fig3_encoding_specificity.png")
    plt.close(fig)
    print("  Saved fig3_encoding_specificity")


def fig4_fan_effect(results):
    """
    Figure 4: Fan effect — MRR vs fan size for both systems.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for sys_name, color, marker in [
        ("BM25", COLORS["bm25"], "o"),
        ("Dense", COLORS["dense"], "s"),
    ]:
        r = results[(sys_name, "Fan Effect")]
        fans = []
        mrrs = []
        for cond, m in sorted(r.conditions.items()):
            fan = int(cond.split("_")[1])
            fans.append(fan)
            mrrs.append(m["mrr"])

        ax.plot(fans, mrrs, f"-{marker}", color=color, label=sys_name,
                linewidth=2, markersize=7, markeredgecolor="white", markeredgewidth=1)

    ax.set_xlabel("Fan Size (documents per entity)")
    ax.set_ylabel("Mean Reciprocal Rank")
    ax.set_title("Figure 4: Fan Effect — Retrieval Accuracy vs. Associative Fan",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    ax.set_xticks([1, 2, 5, 10])
    ax.set_ylim(0, 1.1)

    # Add human prediction arrow
    ax.annotate("Human-like:\ndecreasing MRR",
                xy=(7, 0.5), fontsize=8, color="#666", style="italic",
                ha="center")

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig4_fan_effect.pdf")
    fig.savefig(f"{OUTDIR}/fig4_fan_effect.png")
    plt.close(fig)
    print("  Saved fig4_fan_effect")


def fig5_interference(results):
    """
    Figure 5: Interference — RI vs PI vs Control for both systems.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

    conditions = ["control", "pi_test", "ri_test"]
    labels = ["Control\n(no interference)", "Proactive\nInterference", "Retroactive\nInterference"]

    for ax, sys_name, title in [
        (ax1, "BM25", "BM25"),
        (ax2, "Dense", "Dense Retrieval"),
    ]:
        r = results[(sys_name, "Interference")]

        # Average control phases
        ctrl_mrr = np.mean([
            r.conditions.get("control_phase1", {}).get("mrr", 0),
            r.conditions.get("control_phase2", {}).get("mrr", 0),
        ])
        pi_mrr = r.conditions.get("pi_test", {}).get("mrr", 0)
        ri_mrr = r.conditions.get("ri_test", {}).get("mrr", 0)
        mrrs = [ctrl_mrr, pi_mrr, ri_mrr]

        colors_bar = [COLORS["control"], COLORS["dense"], COLORS["highlight"]]
        x = np.arange(len(conditions))
        bars = ax.bar(x, mrrs, width=0.6, color=colors_bar, alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, mrrs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Mean Reciprocal Rank" if ax == ax1 else "")
        ax.set_title(title, fontsize=10, fontweight="bold")

    fig.suptitle("Figure 5: Proactive and Retroactive Interference",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig5_interference.pdf")
    fig.savefig(f"{OUTDIR}/fig5_interference.png")
    plt.close(fig)
    print("  Saved fig5_interference")


def fig6_cas_summary(results):
    """
    Figure 6: Overall CAS ranking with per-module breakdown.
    """
    modules = ["Encoding Specificity", "Fan Effect", "Interference",
               "Serial Position", "RIF"]
    systems = ["BM25", "Dense"]
    colors_sys = [COLORS["bm25"], COLORS["dense"]]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(modules))
    width = 0.35

    for i, (sys_name, color) in enumerate(zip(systems, colors_sys)):
        scores = [results[(sys_name, m)].cognitive_alignment_score for m in modules]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=sys_name, color=color,
                      alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, scores):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Mean CAS lines — offset labels to avoid overlap
    mean_offsets = [0.04, -0.06]  # first above line, second below
    alignments = ["bottom", "top"]
    for i, (sys_name, color) in enumerate(zip(systems, colors_sys)):
        mean_cas = np.mean([results[(sys_name, m)].cognitive_alignment_score for m in modules])
        ax.axhline(y=mean_cas, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(len(modules) - 0.5, mean_cas + mean_offsets[i],
                f"{sys_name} Mean={mean_cas:.3f}", fontsize=8, color=color,
                ha="right", va=alignments[i])

    ax.set_xticks(x)
    ax.set_xticklabels(["Encoding\nSpecificity", "Fan\nEffect", "Interference",
                         "Serial\nPosition", "Retrieval-Induced\nForgetting"],
                        fontsize=8)
    ax.set_ylabel("Cognitive Alignment Score")
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Figure 6: Cognitive Alignment Score Summary",
                 fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig6_cas_summary.pdf")
    fig.savefig(f"{OUTDIR}/fig6_cas_summary.png")
    plt.close(fig)
    print("  Saved fig6_cas_summary")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Running all benchmarks...")
    results = run_all_benchmarks()

    print("\nGenerating figures...")
    fig1_overview_schematic()
    fig2_heatmap(results)
    fig3_encoding_specificity(results)
    fig4_fan_effect(results)
    fig5_interference(results)
    fig6_cas_summary(results)

    print(f"\nAll figures saved to {OUTDIR}/")
    print("Files generated:")
    for f in sorted(os.listdir(OUTDIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()

# plot_results.py
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from config import BG, GRID, PANEL, RESULTS_FILENAME, TXT, config_by_name, config_names

def pub_style(ax, title, xlabel="Step", ylabel=""):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=TXT)
    ax.grid(True, color=GRID, lw=0.4, linestyle="--")
    ax.set_title(title, color=TXT, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=TXT)
    if ylabel:
        ax.set_ylabel(ylabel, color=TXT)

def savepub(fig, out_path: Path):
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    base_dir = Path(".")
    results_file = base_dir / "ResidualExperiment" / "results" / RESULTS_FILENAME
    fig_dir = base_dir / "ResidualExperiment" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file) as f:
        all_results = json.load(f)

    names = [n for n in config_names() if n in all_results]
    colors = {n: config_by_name(n)["color"] for n in names}

    # Figure 1: Train / Val losses
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_facecolor(BG)
    for ax, key, fk, title in [
        (axes[0], "train_losses", "final_train", "Training Loss"),
        (axes[1], "val_losses", "final_val", "Validation Loss"),
    ]:
        pub_style(ax, f"Figure 1: {title}", ylabel="Loss")
        for n in names:
            res = all_results[n]
            ax.plot(res["iters_log"], res[key], color=colors[n], lw=2, label=f'{n} → {res[fk]:.4f}')
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    fig.suptitle("Figure 1: Train/Val Loss | OpenWebText", color=TXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    savepub(fig, fig_dir / "PUB_fig1_loss_curves.png")

    # Figure 2: Generalisation gap
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    pub_style(ax, "Figure 2: Generalisation Gap (Val − Train)", ylabel="Gap")
    ax.axhline(0, color="white", lw=0.8, linestyle="--", alpha=0.3)
    for n in names:
        res = all_results[n]
        gap = [v - t for v, t in zip(res["val_losses"], res["train_losses"])]
        ax.plot(res["iters_log"], gap, color=colors[n], lw=2, marker="o", ms=3, label=f'{n} final={gap[-1]:.4f}')
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    plt.tight_layout()
    savepub(fig, fig_dir / "PUB_fig2_gen_gap.png")

    # Figure 3: Summary bars
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(BG)
    for ax, metric, label, title in [
        (axes[0], "final_val", "Val Loss", "Final Validation Loss"),
        (axes[1], "final_ppl", "PPL", "Final Perplexity"),
        (axes[2], "final_train", "Train Loss", "Final Training Loss"),
    ]:
        pub_style(ax, title, xlabel="Config", ylabel=label)
        vals = [all_results[n][metric] for n in names]
        bars = ax.bar(range(len(names)), vals, color=[colors[n] for n in names], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.3f}",
                ha="center",
                color=TXT,
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], color=TXT, fontsize=9)
    fig.suptitle("Figure 3: Summary Results | OpenWebText", color=TXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    savepub(fig, fig_dir / "PUB_fig3_summary.png")

    # Figure 4: Gradient flow
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6))
    fig.patch.set_facecolor(BG)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        pub_style(ax, name.replace("_", " "), xlabel="Layer", ylabel="L2 Norm")
        latest = all_results[name]["grad_norm_history"][-1]
        layers = sorted(latest.keys(), key=int)
        x = np.arange(len(layers))
        ax.bar(x - 0.2, [latest[l]["fc1"] for l in layers], 0.4, color=colors[name], alpha=0.85, label="fc1")
        ax.bar(x + 0.2, [latest[l]["fc2"] for l in layers], 0.4, color="#888", alpha=0.85, label="fc2")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], color=TXT, fontsize=8)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=8)
    fig.suptitle("Figure 4: Gradient Flow per FFN Layer", color=TXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    savepub(fig, fig_dir / "PUB_fig4_gradient_flow.png")

    # Figure 5: Hidden norms
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    pub_style(ax, "Figure 5: Hidden-State L2 Norm by Depth", xlabel="Layer", ylabel="Mean ||h||₂")
    for name in names:
        res = all_results[name]
        for i, snap in enumerate(res["hidden_norm_history"][-5:]):
            lyrs = sorted(snap.keys(), key=int)
            ax.plot(
                range(len(lyrs)),
                [snap[l] for l in lyrs],
                color=colors[name],
                alpha=0.3 + 0.14 * i,
                lw=1.5,
                marker="o",
                ms=5,
            )
        snap = res["hidden_norm_history"][-1]
        lyrs = sorted(snap.keys(), key=int)
        ax.plot(
            range(len(lyrs)),
            [snap[l] for l in lyrs],
            color=colors[name],
            lw=3,
            marker="o",
            ms=7,
            label=name.replace("_", " "),
        )
    ax.set_xticks(range(len(lyrs)))
    ax.set_xticklabels([f"L{l}" for l in lyrs], color=TXT, fontsize=9)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=10)
    plt.tight_layout()
    savepub(fig, fig_dir / "PUB_fig5_hidden_norms.png")

    print("\nFinal Results Summary:")
    print(f'{"Config":<22} | {"Val Loss":>10} | {"PPL":>8} | {"Train":>10}')
    print("-" * 58)
    for name in names:
        res = all_results[name]
        print(f'{name:<22} | {res["final_val"]:>10.4f} | {res["final_ppl"]:>8.1f} | {res["final_train"]:>10.4f}')

if __name__ == "__main__":
    main()
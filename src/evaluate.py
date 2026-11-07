"""Evaluation Script for Humour Classification Strategies.

Computes comprehensive classification metrics (Accuracy, Precision, Recall, F1-Score,
Confusion Matrix) across all evaluated strategies (zero_shot, few_shot, cot_style).
Explicitly handles and tracks parse/API failures (pred_label = -1).
Saves structured metric tables and visual comparison charts to results/.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_strategy(csv_path: str, strategy_name: str) -> Dict[str, Any]:
    """Calculate detailed metrics for a single prediction CSV file."""
    if not os.path.exists(csv_path):
        return {
            "strategy": strategy_name,
            "status": "file_not_found",
            "total_rows": 0,
            "valid_rows": 0,
            "failed_rows": 0,
            "failure_rate": 0.0,
        }

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    valid_df = df[df["pred_label"].isin([0, 1])].copy()
    valid_rows = len(valid_df)
    failed_rows = total_rows - valid_rows
    failure_rate = (failed_rows / total_rows * 100) if total_rows > 0 else 0.0

    metrics: Dict[str, Any] = {
        "strategy": strategy_name,
        "status": "completed",
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "failed_rows": failed_rows,
        "failure_rate": round(failure_rate, 2),
    }

    if valid_rows > 0:
        y_true = valid_df["true_label"].astype(int)
        y_pred = valid_df["pred_label"].astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Class 0 specific
        prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

        # Strict accuracy over all rows (treating failures as incorrect)
        correct_count = int((valid_df["true_label"] == valid_df["pred_label"]).sum())
        strict_acc = float(correct_count / total_rows) if total_rows > 0 else 0.0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel()) if cm.shape == (2, 2) else (0, 0, 0, 0)

        metrics.update({
            "accuracy_valid": round(acc, 4),
            "accuracy_strict": round(strict_acc, 4),
            "precision_humour": round(prec, 4),
            "recall_humour": round(rec, 4),
            "f1_humour": round(f1, 4),
            "precision_non_humour": round(prec_0, 4),
            "recall_non_humour": round(rec_0, 4),
            "f1_non_humour": round(f1_0, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "correct_count": correct_count,
        })
    else:
        metrics.update({
            "accuracy_valid": None,
            "accuracy_strict": 0.0,
            "precision_humour": None,
            "recall_humour": None,
            "f1_humour": None,
            "precision_non_humour": None,
            "recall_non_humour": None,
            "f1_non_humour": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "correct_count": 0,
        })

    return metrics


def plot_metrics(summary_df: pd.DataFrame, output_dir: str = "results/figures"):
    """Generate and save publication-quality visual plots."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Pipeline Execution Status
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    strats = summary_df["strategy"].tolist()
    valids = summary_df["valid_rows"].tolist()
    faileds = summary_df["failed_rows"].tolist()
    x = np.arange(len(strats))
    width = 0.55

    ax.bar(x, valids, width, label="Valid Predictions (0/1)", color="#2563eb")
    ax.bar(x, faileds, width, bottom=valids, label="Failures / Quota (-1)", color="#ef4444")
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_title("Pipeline Completion Status by Strategy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strats, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)

    for i in range(len(strats)):
        tot = valids[i] + faileds[i]
        if valids[i] > 0:
            ax.text(i, valids[i] / 2, f"{valids[i]}", ha="center", va="center", color="white", fontweight="bold")
        if faileds[i] > 0:
            ax.text(i, valids[i] + faileds[i] / 2, f"{faileds[i]}", ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pipeline_execution_status.png"))
    plt.close()

    # 2. Performance Comparison on Valid Rows
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    width = 0.25
    x = np.arange(len(metrics_names))
    colors = ["#10b981", "#3b82f6", "#8b5cf6"]

    for idx, row in summary_df.iterrows():
        strat = row["strategy"]
        if row["valid_rows"] > 0 and row["accuracy_valid"] is not None:
            vals = [
                row["accuracy_valid"] * 100,
                row["precision_humour"] * 100,
                row["recall_humour"] * 100,
                row["f1_humour"] * 100,
            ]
        else:
            vals = [0.0, 0.0, 0.0, 0.0]

        offset = (idx - 1) * width
        rects = ax.bar(x + offset, vals, width, label=f"{strat} (N={row['valid_rows']})", color=colors[idx % len(colors)])
        for r in rects:
            h = r.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
            else:
                ax.annotate("N/A", xy=(r.get_x() + r.get_width() / 2, 0), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#6b7280")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Metrics Comparison (Valid Predictions)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    plt.close()

    # 3. Side-by-side Confusion Matrices
    n_strats = len(summary_df)
    fig, axes = plt.subplots(1, n_strats, figsize=(4.5 * n_strats, 4), dpi=300)
    if n_strats == 1:
        axes = [axes]

    cm_cmaps = ["Blues", "Greens", "Purples"]
    for idx, row in summary_df.iterrows():
        ax = axes[idx]
        cm_dict = row["confusion_matrix"]
        cm_array = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
        sns.heatmap(cm_array, annot=True, fmt="d", cmap=cm_cmaps[idx % len(cm_cmaps)], ax=ax, cbar=False,
                    annot_kws={"size": 13, "fontweight": "bold"},
                    xticklabels=["Not Humour (0)", "Humour (1)"],
                    yticklabels=["Not Humour (0)", "Humour (1)"])
        title = f"{row['strategy']}\n(Valid: {row['valid_rows']}/{row['total_rows']})"
        if row["accuracy_valid"] is not None:
            title += f" | Acc: {row['accuracy_valid']*100:.1f}%"
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=9, fontweight="bold")
        ax.set_ylabel("True", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate humour detection predictions.")
    parser.add_argument("--pred_dir", type=str, default="results/predictions", help="Predictions directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save metric outputs")
    parser.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot", "cot_style"], help="Strategies")
    args = parser.parse_args()

    results_list: List[Dict[str, Any]] = []

    print("\n" + "=" * 78)
    print("HUMOUR DETECTION MODEL EVALUATION PIPELINE")
    print("=" * 78)

    for strat in args.strategies:
        csv_file = os.path.join(args.pred_dir, f"{strat}.csv")
        metrics = evaluate_strategy(csv_file, strat)
        results_list.append(metrics)

    summary_df = pd.DataFrame(results_list)

    # Console Output
    print("\n" + "-" * 78)
    print("SUMMARY METRICS TABLE:")
    print("-" * 78)
    display_cols = [
        "strategy", "total_rows", "valid_rows", "failed_rows",
        "accuracy_valid", "precision_humour", "recall_humour", "f1_humour", "macro_f1"
    ]
    print(summary_df[display_cols].to_string(index=False))

    # Save to JSON and CSV
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\n[OK] Metrics saved to: {json_path}")
    print(f"[OK] Metrics CSV saved to: {csv_path}")

    # Plot visual artifacts
    fig_dir = os.path.join(args.output_dir, "figures")
    plot_metrics(summary_df, output_dir=fig_dir)
    print(f"[OK] Visual plots saved to: {fig_dir}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

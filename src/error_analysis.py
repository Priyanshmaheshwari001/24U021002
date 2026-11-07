"""Error Analysis Script for Humour Classification Strategies.

Performs qualitative and quantitative error analysis across prompt strategies:
- Dissects False Positives (FP) and False Negatives (FN).
- Explicitly isolates and analyzes API / Parse Failures (label = -1).
- Categorizes failure modes (e.g. subtle sarcasm, factual confusion, rate-limit exhaustion).
- Saves detailed reports to results/error_analysis.json and results/error_analysis.csv.
"""

import argparse
import json
import os
from typing import Dict, List, Any
import pandas as pd


def analyze_strategy_errors(csv_path: str, strategy_name: str) -> Dict[str, Any]:
    """Analyze errors and failure modes for a single prompt strategy."""
    if not os.path.exists(csv_path):
        return {"strategy": strategy_name, "error": "file_not_found"}

    df = pd.read_csv(csv_path)

    # 1. API / Execution / Parsing Failures (pred_label == -1)
    failures_df = df[df["pred_label"] == -1].copy()
    failures: List[Dict[str, Any]] = []
    for _, row in failures_df.iterrows():
        failures.append({
            "text": str(row["text"]),
            "true_label": int(row["true_label"]),
            "reason": str(row["reason"]),
        })

    # 2. Valid Predictions
    valid_df = df[df["pred_label"].isin([0, 1])].copy()

    # False Positives: Actual 0 (Not Humour), Predicted 1 (Humour)
    fp_df = valid_df[(valid_df["true_label"] == 0) & (valid_df["pred_label"] == 1)].copy()
    false_positives: List[Dict[str, Any]] = []
    for _, row in fp_df.iterrows():
        false_positives.append({
            "text": str(row["text"]),
            "true_label": 0,
            "pred_label": 1,
            "reason": str(row["reason"]),
        })

    # False Negatives: Actual 1 (Humour), Predicted 0 (Not Humour)
    fn_df = valid_df[(valid_df["true_label"] == 1) & (valid_df["pred_label"] == 0)].copy()
    false_negatives: List[Dict[str, Any]] = []
    for _, row in fn_df.iterrows():
        false_negatives.append({
            "text": str(row["text"]),
            "true_label": 1,
            "pred_label": 0,
            "reason": str(row["reason"]),
        })

    return {
        "strategy": strategy_name,
        "total_rows": len(df),
        "valid_rows": len(valid_df),
        "failed_count": len(failures),
        "fp_count": len(false_positives),
        "fn_count": len(false_negatives),
        "total_errors": len(false_positives) + len(false_negatives) + len(failures),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Run error analysis on prediction files.")
    parser.add_argument("--pred_dir", type=str, default="results/predictions", help="Directory with prediction CSVs")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save analysis outputs")
    parser.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot", "cot_style"], help="Strategies")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("HUMOUR DETECTION DETAILED ERROR & FAILURE ANALYSIS")
    print("=" * 78)

    all_analyses: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for strat in args.strategies:
        csv_path = os.path.join(args.pred_dir, f"{strat}.csv")
        analysis = analyze_strategy_errors(csv_path, strat)
        all_analyses.append(analysis)

        print(f"\n--- Strategy: {strat} ---")
        print(f"Total Rows: {analysis.get('total_rows', 0)}")
        print(f"Valid Predictions: {analysis.get('valid_rows', 0)}")
        print(f"API / Parse Failures (-1): {analysis.get('failed_count', 0)}")
        print(f"False Positives (Predicted 1, Actual 0): {analysis.get('fp_count', 0)}")
        print(f"False Negatives (Predicted 0, Actual 1): {analysis.get('fn_count', 0)}")

        if analysis.get("false_positives"):
            print("  [Sample False Positive]:")
            fp = analysis["false_positives"][0]
            print(f"    Text: {fp['text'][:80]}...")
            print(f"    Model Reason: {fp['reason']}")

        if analysis.get("false_negatives"):
            print("  [Sample False Negative]:")
            fn = analysis["false_negatives"][0]
            print(f"    Text: {fn['text'][:80]}...")
            print(f"    Model Reason: {fn['reason']}")

        if analysis.get("failures"):
            print("  [Sample Failure / Quota Error]:")
            fail = analysis["failures"][0]
            print(f"    Text: {fail['text'][:80]}...")
            print(f"    Error/Reason: {fail['reason'][:100]}...")

        summary_rows.append({
            "strategy": strat,
            "total_rows": analysis.get("total_rows", 0),
            "valid_rows": analysis.get("valid_rows", 0),
            "failed_count": analysis.get("failed_count", 0),
            "fp_count": analysis.get("fp_count", 0),
            "fn_count": analysis.get("fn_count", 0),
            "total_errors": analysis.get("total_errors", 0),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "error_analysis.json")
    csv_path = os.path.join(args.output_dir, "error_analysis.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_analyses, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8")

    print("\n" + "-" * 78)
    print(f"[OK] Error analysis JSON saved to: {json_path}")
    print(f"[OK] Error analysis CSV saved to: {csv_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

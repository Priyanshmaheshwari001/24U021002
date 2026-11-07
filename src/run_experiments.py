"""Experiment Runner for Humour Detection Prompt Strategies.

Evaluates prompt strategies ('zero_shot', 'few_shot', 'cot_style') on the
evaluation sample dataset (data/processed/eval_sample.csv) using Gemini LLM.

Features:
- Live tqdm progress bar for inference.
- Saves predictions after each strategy to results/predictions/<strategy>.csv with
  exact columns: text, true_label, pred_label, reason.
- Resumes execution by checking existing output CSVs and the local SQLite cache.
- Accepts --limit N to run on a small subset first for validation.
- Explicitly tracks, counts, and reports JSON/label parse failures (label = -1).
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

# Ensure local package imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client import classify_humour, get_api_key, MODEL_NAME
from src.prompts import PROMPT_TEMPLATES, get_prompt

# Ensure utf-8 output encoding in Windows shells
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def run_strategy(
    strategy_name: str,
    eval_df: pd.DataFrame,
    output_dir: str = "results/predictions",
    model: str = MODEL_NAME,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Run evaluation for a single prompt strategy."""
    prompt_fn = PROMPT_TEMPLATES[strategy_name]
    output_file = os.path.join(output_dir, f"{strategy_name}.csv")
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing partial or completed results to support resuming
    existing_results: Dict[str, Dict] = {}
    if os.path.exists(output_file):
        try:
            prev_df = pd.read_csv(output_file)
            required_cols = {"text", "true_label", "pred_label", "reason"}
            if required_cols.issubset(set(prev_df.columns)):
                for _, row in prev_df.iterrows():
                    # Only reuse non-failed predictions
                    if int(row["pred_label"]) != -1:
                        existing_results[str(row["text"])] = {
                            "true_label": int(row["true_label"]),
                            "pred_label": int(row["pred_label"]),
                            "reason": str(row["reason"]),
                        }
                print(f"[{strategy_name}] Found existing output file with {len(existing_results)} valid completed rows. Resuming...")
        except Exception as e:
            print(f"[{strategy_name}] Warning: Could not read existing output file ({e}). Starting fresh.")

    results: List[Dict] = []
    failed_parses_count = 0
    cache_hits_count = 0
    api_calls_count = 0

    print(f"\n{'=' * 75}")
    print(f"RUNNING STRATEGY: '{strategy_name}' (Model: {model})")
    print(f"Evaluating {len(eval_df)} samples...")
    print(f"{'=' * 75}")

    progress_bar = tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Strategy: {strategy_name}", unit="post")

    for idx, row in progress_bar:
        raw_text = str(row["text"])
        # Prefer clean_text for prompt feeding if available, while keeping raw_text for storage
        text_for_model = str(row["clean_text"]) if "clean_text" in row and pd.notna(row["clean_text"]) else raw_text
        true_label = int(row["label"])

        # Check if already present in previously saved file
        if raw_text in existing_results and use_cache:
            prev = existing_results[raw_text]
            results.append({
                "text": raw_text,
                "true_label": true_label,
                "pred_label": prev["pred_label"],
                "reason": prev["reason"],
            })
            cache_hits_count += 1
            continue

        try:
            # classify_humour handles internal SQLite caching and exponential backoff
            classification = classify_humour(
                text=text_for_model,
                prompt_template=prompt_fn,
                model=model,
                use_cache=use_cache,
            )

            pred_label = int(classification.get("label", -1))
            reason = str(classification.get("reason", "")).strip()

            if classification.get("cached", False):
                cache_hits_count += 1
            else:
                api_calls_count += 1

            if pred_label == -1:
                failed_parses_count += 1

            results.append({
                "text": raw_text,
                "true_label": true_label,
                "pred_label": pred_label,
                "reason": reason,
            })

        except Exception as exc:
            # Handle unrecoverable API/network errors
            print(f"\n[ERROR] Exception processing row {idx}: {exc}")
            failed_parses_count += 1
            results.append({
                "text": raw_text,
                "true_label": true_label,
                "pred_label": -1,
                "reason": f"Execution error: {exc}",
            })

            # Check for invalid API key and halt if detected
            msg = str(exc).lower()
            if "invalid api key" in msg or "api key not valid" in msg or "no api key was provided" in msg:
                print("\n" + "!" * 75)
                print("[FATAL ERROR] API key is missing or invalid! Halting experiment.")
                print("!" * 75)
                break

        # Progressive save to ensure no loss of progress
        pd.DataFrame(results, columns=["text", "true_label", "pred_label", "reason"]).to_csv(
            output_file, index=False, encoding="utf-8"
        )

    # Save predictions DataFrame
    results_df = pd.DataFrame(results, columns=["text", "true_label", "pred_label", "reason"])
    results_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nSaved predictions to: {output_file}")

    # Summary Statistics
    total_evaluated = len(results_df)
    valid_predictions = results_df[results_df["pred_label"].isin([0, 1])]
    n_valid = len(valid_predictions)
    n_failed = (results_df["pred_label"] == -1).sum()

    print(f"\n--- Strategy Summary: {strategy_name} ---")
    print(f"Total Rows Evaluated: {total_evaluated}")
    print(f"Cache Hits: {cache_hits_count} | Live API Calls: {api_calls_count}")
    print(f"Parse Failures (pred_label = -1): {n_failed} ({ (n_failed / total_evaluated * 100) if total_evaluated > 0 else 0:.2f}%)")

    if n_valid > 0:
        correct = (valid_predictions["true_label"] == valid_predictions["pred_label"]).sum()
        accuracy = (correct / n_valid) * 100
        print(f"Accuracy on Valid Rows: {correct}/{n_valid} ({accuracy:.2f}%)")
        print("Predicted Class Counts (on valid rows):")
        for lbl, count in valid_predictions["pred_label"].value_counts().items():
            desc = "Funny (1)" if lbl == 1 else "Not Funny (0)"
            print(f"  {desc}: {count} ({count / n_valid * 100:.2f}%)")
    else:
        print("No valid predictions obtained.")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run humour classification prompt experiments with Gemini API.")
    parser.add_argument(
        "--eval_path",
        type=str,
        default="data/processed/eval_sample.csv",
        help="Path to evaluation dataset CSV (default: data/processed/eval_sample.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/predictions",
        help="Directory to save prediction CSV files (default: results/predictions)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "few_shot", "cot_style"],
        choices=["zero_shot", "few_shot", "cot_style"],
        help="Strategies to evaluate (default: all three)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N rows (useful for initial smoke testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Gemini model identifier (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable SQLite caching (forces live API requests)",
    )

    args = parser.parse_args()

    # Pre-flight Check: API Key
    try:
        get_api_key()
    except Exception as exc:
        print("\n" + "!" * 75)
        print("[API KEY ERROR] Action Required:")
        print(exc)
        print("!" * 75 + "\n")
        sys.exit(1)

    if not os.path.exists(args.eval_path):
        print(f"[ERROR] Evaluation file not found at: {args.eval_path}")
        print("Please run 'python src/prepare_data.py' first.")
        sys.exit(1)

    eval_df = pd.read_csv(args.eval_path)
    if args.limit is not None and args.limit > 0:
        print(f"[INFO] Limiting evaluation to first {args.limit} rows.")
        eval_df = eval_df.head(args.limit)

    print(f"\nLoaded evaluation dataset: {len(eval_df)} rows from {args.eval_path}")
    print(f"Selected strategies: {args.strategies}")
    print(f"Model: {args.model}")

    for strategy in args.strategies:
        run_strategy(
            strategy_name=strategy,
            eval_df=eval_df,
            output_dir=args.output_dir,
            model=args.model,
            use_cache=not args.no_cache,
        )

    print("\n" + "=" * 75)
    print("ALL REQUESTED EXPERIMENTS COMPLETED.")
    print("=" * 75)


if __name__ == "__main__":
    main()

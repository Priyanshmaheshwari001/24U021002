"""Data Preparation and Preprocessing Script for Humour Detection.

Performs light cleaning on the raw HaHackathon humour dataset:
- Removes URLs
- Replaces @mentions with '@user'
- Converts emojis to descriptive text tags using emoji.demojize
- Collapses extra whitespace and trims leading/trailing spaces
- Keeps original case, punctuation, and stopwords (preserves tone for LLMs)
- Drops empty strings and duplicate entries
- Generates a balanced evaluation sample (default 300 funny + 300 not funny)
- Generates a disjoint few-shot pool (default 10 funny + 10 not funny)
"""

import argparse
import os
import re
import sys
import emoji
import pandas as pd

# Ensure proper utf-8 encoding for standard output (especially on Windows)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def clean_text(text: str) -> str:
    """Apply light cleaning suitable for LLM prompting.

    Preserves original casing, punctuation, and stopwords while standardizing
    URLs, user handles, emojis, and whitespace.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 2. Replace @mentions with '@user'
    text = re.sub(r"@\w+", "@user", text)

    # 3. Demojize emojis (converts unicode emojis to textual representations like :smile:)
    text = emoji.demojize(text)

    # 4. Collapse multiple whitespace characters into a single space and strip edges
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_dataset(
    raw_path: str = "data/raw/hahackathon_train.csv",
    processed_dir: str = "data/processed",
    n_per_class: int = 300,
    fewshot_per_class: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, and split the raw dataset into eval sample and few-shot pool."""
    print("=" * 70)
    print("STEP 1: Loading Raw Dataset")
    print("=" * 70)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    raw_df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset from: {raw_path}")
    print(f"Raw shape: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")

    # Validate required columns
    text_col = "text"
    label_col = "is_humor"
    if text_col not in raw_df.columns:
        raise ValueError(f"Expected text column '{text_col}' not found in {list(raw_df.columns)}")
    if label_col not in raw_df.columns:
        raise ValueError(f"Expected label column '{label_col}' not found in {list(raw_df.columns)}")

    df = raw_df[[text_col, label_col]].copy()
    df.rename(columns={label_col: "label"}, inplace=True)

    print("\n" + "=" * 70)
    print("STEP 2: Light Cleaning")
    print("=" * 70)
    print("Applying cleaning: removing URLs, replacing @mentions, demojizing emojis, stripping whitespace...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Drop null / empty texts
    initial_count = len(df)
    df = df[df["clean_text"].str.strip() != ""].copy()
    df = df.dropna(subset=["clean_text", "label"])
    non_empty_count = len(df)
    dropped_empty = initial_count - non_empty_count
    print(f"Dropped {dropped_empty} empty rows after cleaning.")

    # Drop duplicates on clean_text
    df = df.drop_duplicates(subset=["clean_text"]).copy()
    dropped_duplicates = non_empty_count - len(df)
    print(f"Dropped {dropped_duplicates} duplicate clean_text rows.")
    print(f"Remaining valid rows after cleaning: {len(df):,}")

    # Ensure label is integer
    df["label"] = df["label"].astype(int)

    # Verify both classes exist
    class_counts = df["label"].value_counts().to_dict()
    print(f"Cleaned dataset class distribution: {class_counts}")
    if 0 not in class_counts or 1 not in class_counts:
        raise ValueError(f"Dataset missing required binary classes! Found classes: {class_counts}")

    print("\n" + "=" * 70)
    print(f"STEP 3: Creating Evaluation Sample ({n_per_class} per class)")
    print("=" * 70)
    df_class_1 = df[df["label"] == 1]
    df_class_0 = df[df["label"] == 0]

    if len(df_class_1) < (n_per_class + fewshot_per_class):
        raise ValueError(
            f"Insufficient class 1 samples ({len(df_class_1)}) for eval ({n_per_class}) + few-shot ({fewshot_per_class})"
        )
    if len(df_class_0) < (n_per_class + fewshot_per_class):
        raise ValueError(
            f"Insufficient class 0 samples ({len(df_class_0)}) for eval ({n_per_class}) + few-shot ({fewshot_per_class})"
        )

    # Sample evaluation dataset
    eval_1 = df_class_1.sample(n=n_per_class, random_state=random_state)
    eval_0 = df_class_0.sample(n=n_per_class, random_state=random_state)

    eval_sample = pd.concat([eval_1, eval_0]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    eval_sample = eval_sample[["text", "clean_text", "label"]]

    print("\n" + "=" * 70)
    print(f"STEP 4: Creating Disjoint Few-Shot Pool ({fewshot_per_class} per class)")
    print("=" * 70)
    # Remaining candidates strictly excluding eval samples
    rem_1 = df_class_1.drop(eval_1.index)
    rem_0 = df_class_0.drop(eval_0.index)

    fewshot_1 = rem_1.sample(n=fewshot_per_class, random_state=random_state)
    fewshot_0 = rem_0.sample(n=fewshot_per_class, random_state=random_state)

    fewshot_pool = pd.concat([fewshot_1, fewshot_0]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    fewshot_pool = fewshot_pool[["text", "clean_text", "label"]]

    # Verification: assert 0 overlap between evaluation sample and few-shot pool
    eval_texts = set(eval_sample["clean_text"])
    fewshot_texts = set(fewshot_pool["clean_text"])
    overlap = eval_texts.intersection(fewshot_texts)
    assert len(overlap) == 0, f"Error: Found {len(overlap)} overlapping samples between eval and few-shot datasets!"
    print(f"Verification successful: 0 overlap between eval sample and few-shot pool.")

    print("\n" + "=" * 70)
    print("STEP 5: Saving Processed Datasets")
    print("=" * 70)
    os.makedirs(processed_dir, exist_ok=True)
    eval_path = os.path.join(processed_dir, "eval_sample.csv")
    fewshot_path = os.path.join(processed_dir, "fewshot_pool.csv")

    eval_sample.to_csv(eval_path, index=False, encoding="utf-8")
    fewshot_pool.to_csv(fewshot_path, index=False, encoding="utf-8")

    print(f"Saved eval sample to: {eval_path}")
    print(f"Saved few-shot pool to: {fewshot_path}")

    print("\n" + "=" * 70)
    print("STEP 6: Dataset Sizes & Class Counts")
    print("=" * 70)

    def print_stats(name: str, sample_df: pd.DataFrame):
        total = len(sample_df)
        counts = sample_df["label"].value_counts().sort_index()
        print(f"\n--- {name} ---")
        print(f"Total Rows: {total}")
        print(f"Columns: {list(sample_df.columns)}")
        for label, count in counts.items():
            pct = (count / total) * 100
            desc = "Funny (1)" if label == 1 else "Not Funny (0)"
            print(f"  Class {desc}: {count} ({pct:.2f}%)")

    print_stats("Evaluation Sample (eval_sample.csv)", eval_sample)
    print_stats("Few-Shot Pool (fewshot_pool.csv)", fewshot_pool)

    return eval_sample, fewshot_pool


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and clean dataset for humour detection")
    parser.add_argument(
        "--raw_path",
        type=str,
        default="data/raw/hahackathon_train.csv",
        help="Path to raw dataset CSV file (default: data/raw/hahackathon_train.csv)",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed datasets (default: data/processed)",
    )
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=300,
        help="Number of samples per class for evaluation sample (default: 300 -> total 600)",
    )
    parser.add_argument(
        "--fewshot_per_class",
        type=int,
        default=10,
        help="Number of samples per class for few-shot pool (default: 10 -> total 20)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        raw_path=args.raw_path,
        processed_dir=args.processed_dir,
        n_per_class=args.n_per_class,
        fewshot_per_class=args.fewshot_per_class,
        random_state=args.random_state,
    )

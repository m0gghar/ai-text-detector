"""Kaggle Dataset Ingestion & Processing Script.
Loads the 'shanegerami/ai-vs-human-text' (AI_Human.csv) dataset (~500,000 rows, 1.11 GB),
performs balanced stratification, text cleaning, and updates train/val/test splits.
"""

import os
import sys
import glob
import zipfile
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup project root in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text, get_word_count

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DEFAULT_CSV_PATH = os.path.join(RAW_DIR, "AI_Human.csv")


def find_or_extract_kaggle_dataset() -> str:
    """Find AI_Human.csv or extract it if an archive (.zip) exists in data/raw."""
    # 1. Check for AI_Human.csv directly
    if os.path.exists(DEFAULT_CSV_PATH):
        return DEFAULT_CSV_PATH

    # 2. Check for any CSV in data/raw
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if csv_files:
        return csv_files[0]

    # 3. Check for any zip archive in data/raw and extract it
    zip_files = glob.glob(os.path.join(RAW_DIR, "*.zip"))
    if zip_files:
        zip_path = zip_files[0]
        print(f"Found archive {zip_path}. Extracting to {RAW_DIR}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)
        
        extracted_csvs = glob.glob(os.path.join(RAW_DIR, "*.csv"))
        if extracted_csvs:
            return extracted_csvs[0]

    return None


def process_kaggle_dataset(csv_path: str, sample_size: int = 20000, random_state: int = 42):
    """Process Kaggle AI_Human.csv dataset:
    - Standardizes column names ('text', 'generated' -> 'label')
    - Filters out empty or corrupted rows
    - Takes a balanced stratified sample (e.g. 10,000 Human + 10,000 AI) for optimal training velocity
    - Splits into Train (70%), Validation (15%), Test (15%)
    """
    print(f"\n[1/4] Reading Kaggle dataset from: {csv_path}")
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    # Read CSV
    # The Kaggle dataset columns are: 'text', 'generated'
    df = pd.read_csv(csv_path)
    print(f"Total rows in raw dataset: {len(df):,}")

    # Identify label column
    if "generated" in df.columns:
        df["label"] = df["generated"].astype(int)
    elif "label" in df.columns:
        df["label"] = df["label"].astype(int)
    else:
        raise ValueError(f"Could not find 'generated' or 'label' column in {df.columns.tolist()}")

    # Clean missing / null values
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    human_count = (df["label"] == 0).sum()
    ai_count = (df["label"] == 1).sum()
    print(f"Class distribution in full file -> Human (0): {human_count:,} | AI (1): {ai_count:,}")

    # Subsample if requested (PRD recommendation for fast, robust local training)
    if sample_size and sample_size < len(df):
        print(f"\n[2/4] Sampling {sample_size:,} balanced records ({sample_size//2:,} Human, {sample_size//2:,} AI)...")
        half_sample = sample_size // 2
        df_human = df[df["label"] == 0].sample(n=min(half_sample, human_count), random_state=random_state)
        df_ai = df[df["label"] == 1].sample(n=min(half_sample, ai_count), random_state=random_state)
        df = pd.concat([df_human, df_ai]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        print(f"\n[2/4] Using all {len(df):,} records...")

    # Calculate word counts and clean text
    print("\n[3/4] Preprocessing texts...")
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["word_count"] = df["text"].apply(get_word_count)
    df["source"] = "kaggle_shanegerami"

    # Filter out extremely short corrupt texts (< 10 words)
    df = df[df["word_count"] >= 10].reset_index(drop=True)
    print(f"Valid preprocessed samples: {len(df):,}")

    # Stratified Splits: 70% Train, 15% Val, 15% Test
    print("\n[4/4] Creating stratified splits (70% Train, 15% Val, 15% Test)...")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["label"]
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Successfully saved processed splits:")
    print(f"  - Train: {len(train_df):,} samples -> {train_path}")
    print(f"  - Val:   {len(val_df):,} samples -> {val_path}")
    print(f"  - Test:  {len(test_df):,} samples -> {test_path}")

    return {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest Kaggle AI vs Human Text dataset.")
    parser.add_argument("--csv", type=str, default=None, help="Path to AI_Human.csv")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Number of balanced samples to extract (default: 20000 for balanced high-velocity training). Use 0 for full dataset.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Automatically trigger baseline model training immediately after dataset preparation.",
    )
    args = parser.parse_args()

    csv_path = args.csv or find_or_extract_kaggle_dataset()

    if not csv_path or not os.path.exists(csv_path):
        print("\n" + "=" * 70)
        print("ERROR: Kaggle dataset 'AI_Human.csv' was not found in data/raw/!")
        print("=" * 70)
        print("\nPlease follow these simple steps:")
        print("1. Download the dataset zip from Kaggle:")
        print("   https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text")
        print("2. Extract AI_Human.csv and place it at:")
        print(f"   {DEFAULT_CSV_PATH}")
        print("3. Re-run this script:")
        print("   py -3.12 -m data.load_kaggle_dataset")
        print("=" * 70 + "\n")
        sys.exit(1)

    sample_size = None if args.sample_size == 0 else args.sample_size
    process_kaggle_dataset(csv_path, sample_size=sample_size)

    if args.train:
        print("\n--- Triggering Baseline Model Training on Kaggle Data ---")
        from src.baseline_model import train_baseline_models
        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
        train_baseline_models(train_df, val_df)


if __name__ == "__main__":
    main()

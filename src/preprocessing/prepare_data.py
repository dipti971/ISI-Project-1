"""Main preprocessing pipeline.

Transforms the cleaned CICIDS2017 Friday dataset into ML-ready arrays:
  1. Load friday_clean.csv
  2. Drop non-feature columns (Timestamp, Attempted Category)
  3. Handle infinities → replace with column max finite value
  4. Handle NaN → fill with 0
  5. Merge "Botnet - Attempted" into "Botnet" → 4 classes
  6. Encode labels with LabelEncoder
  7. Feature selection (variance + correlation)
  8. Normalize with StandardScaler
  9. Stratified train/test split (80/20)
 10. Save .npy / .pkl outputs
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path so we can import sibling modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.feature_selection import select_features

# ==========================
# Configuration
# ==========================

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "friday_clean.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Columns to drop (not useful as features)
DROP_COLUMNS = ["Timestamp", "Attempted Category"]

# IP columns to preserve for graph construction (saved separately)
IP_COLUMNS = ["Src IP dec", "Dst IP dec"]

# Label column
LABEL_COLUMN = "Label"

# Label merging map
LABEL_MERGE = {
    "Botnet - Attempted": "Botnet",
}

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    """Load the cleaned dataset."""
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    df = pd.read_csv(INPUT_PATH)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_labels(df):
    """Merge label classes and encode."""
    print("\n" + "=" * 60)
    print("LABEL PROCESSING")
    print("=" * 60)

    # Merge labels
    df[LABEL_COLUMN] = df[LABEL_COLUMN].replace(LABEL_MERGE)
    print("  Merged 'Botnet - Attempted' -> 'Botnet'")

    # Show distribution
    print("\n  Label distribution:")
    counts = df[LABEL_COLUMN].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"    {label:>12s}: {count:>8,d}  ({pct:.1f}%)")

    # Encode
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COLUMN])

    print(f"\n  Encoded {len(le.classes_)} classes:")
    for idx, name in enumerate(le.classes_):
        print(f"    {idx} -> {name}")

    return y, le


def clean_features(df):
    """Drop non-feature columns, handle infinities and NaN."""
    print("\n" + "=" * 60)
    print("FEATURE CLEANING")
    print("=" * 60)

    # Separate IP columns (save for graph construction later)
    ip_data = df[IP_COLUMNS].values.copy()
    print(f"  Preserved IP columns: {IP_COLUMNS}")

    # Drop non-feature columns
    cols_to_drop = [c for c in DROP_COLUMNS + IP_COLUMNS + [LABEL_COLUMN]
                    if c in df.columns]
    df_features = df.drop(columns=cols_to_drop)
    print(f"  Dropped columns: {cols_to_drop}")
    print(f"  Remaining features: {df_features.shape[1]}")

    # Convert all to numeric (coerce errors to NaN)
    df_features = df_features.apply(pd.to_numeric, errors="coerce")

    # Handle infinities: replace with column-wise finite max
    inf_count = np.isinf(df_features.values).sum()
    if inf_count > 0:
        for col in df_features.columns:
            col_data = df_features[col]
            finite_mask = np.isfinite(col_data)
            if not finite_mask.all():
                finite_max = col_data[finite_mask].max() if finite_mask.any() else 0
                df_features[col] = col_data.replace([np.inf, -np.inf], finite_max)
        print(f"  Replaced {inf_count} infinite values with column max")
    else:
        print("  No infinite values found")

    # Handle NaN: fill with 0
    nan_count = df_features.isnull().sum().sum()
    if nan_count > 0:
        df_features = df_features.fillna(0)
        print(f"  Filled {nan_count} NaN values with 0")
    else:
        print("  No NaN values found")

    return df_features, ip_data


def normalize_features(X_train, X_test, feature_names):
    """Apply StandardScaler normalization."""
    print("\n" + "=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  StandardScaler fitted on training data")
    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Test shape:  {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


def save_outputs(X_train, X_test, y_train, y_test,
                 ip_train, ip_test, feature_names, le, scaler):
    """Save all preprocessed outputs."""
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Feature arrays
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    print(f"  X_train.npy  -> {X_train.shape}")
    print(f"  X_test.npy   -> {X_test.shape}")

    # Label arrays
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    print(f"  y_train.npy  -> {y_train.shape}")
    print(f"  y_test.npy   -> {y_test.shape}")

    # IP columns (for graph construction)
    np.save(os.path.join(OUTPUT_DIR, "ip_train.npy"), ip_train)
    np.save(os.path.join(OUTPUT_DIR, "ip_test.npy"), ip_test)
    print(f"  ip_train.npy -> {ip_train.shape}")
    print(f"  ip_test.npy  -> {ip_test.shape}")

    # Feature names
    np.save(os.path.join(OUTPUT_DIR, "feature_names.npy"),
            np.array(feature_names))
    print(f"  feature_names.npy -> {len(feature_names)} features")

    # Label encoder
    le_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"  label_encoder.pkl -> {list(le.classes_)}")

    # Scaler
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  scaler.pkl -> saved")


def main():
    """Run the full preprocessing pipeline."""
    print("\n" + "#" * 60)
    print("#  CICIDS2017 PREPROCESSING PIPELINE")
    print("#" * 60)

    # Step 1: Load
    df = load_data()

    # Step 2: Process labels (merge Botnet classes, encode)
    y, le = handle_labels(df)

    # Step 3: Clean features (drop cols, handle inf/NaN)
    df_features, ip_data = clean_features(df)

    # Step 4: Feature selection (variance + correlation)
    df_selected, feature_names, selection_summary = select_features(
        df_features,
        variance_thresh=0.01,
        correlation_thresh=0.95,
    )

    X = df_selected.values

    # Step 5: Stratified train/test split
    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test, ip_train, ip_test = train_test_split(
        X, y, ip_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"  Train: {X_train.shape[0]:,d} samples ({100 * (1 - TEST_SIZE):.0f}%)")
    print(f"  Test:  {X_test.shape[0]:,d} samples ({100 * TEST_SIZE:.0f}%)")

    # Step 6: Normalize
    X_train, X_test, scaler = normalize_features(X_train, X_test, feature_names)

    # Step 7: Save everything
    save_outputs(X_train, X_test, y_train, y_test,
                 ip_train, ip_test, feature_names, le, scaler)

    # Final summary
    print("\n" + "#" * 60)
    print("#  PIPELINE COMPLETE")
    print("#" * 60)
    print(f"  Classes:         {list(le.classes_)}")
    print(f"  Features:        {len(feature_names)} "
          f"(from {selection_summary['original_count']})")
    print(f"  Train samples:   {X_train.shape[0]:,d}")
    print(f"  Test samples:    {X_test.shape[0]:,d}")
    print(f"  Output dir:      {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

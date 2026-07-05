"""Feature selection helpers.

Two-stage feature selection:
1. Variance thresholding — remove features with near-zero variance.
2. Correlation filtering — remove one feature from each highly correlated pair.
"""

import numpy as np
import pandas as pd


def variance_threshold(df, threshold=0.01):
    """Remove columns whose variance is below *threshold*.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric feature matrix.
    threshold : float
        Minimum variance to keep a feature.

    Returns
    -------
    pd.DataFrame
        DataFrame with low-variance columns dropped.
    list[str]
        Names of the dropped columns.
    """
    variances = df.var()
    low_var_cols = variances[variances < threshold].index.tolist()
    df_filtered = df.drop(columns=low_var_cols)

    print(f"  Variance threshold ({threshold}): dropped {len(low_var_cols)} columns")
    if low_var_cols:
        for col in low_var_cols:
            print(f"    - {col}  (var={variances[col]:.6f})")

    return df_filtered, low_var_cols


def correlation_filter(df, threshold=0.95):
    """Remove one feature from each pair whose Pearson |r| > *threshold*.

    For every correlated pair, the feature appearing later in column order
    is dropped (simple heuristic that keeps earlier / more "fundamental"
    features).

    Parameters
    ----------
    df : pd.DataFrame
        Numeric feature matrix.
    threshold : float
        Maximum allowed absolute correlation.

    Returns
    -------
    pd.DataFrame
        DataFrame with redundant columns dropped.
    list[str]
        Names of the dropped columns.
    """
    corr_matrix = df.corr().abs()

    # Upper triangle mask (exclude diagonal and lower triangle)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    # Find columns where at least one correlation exceeds threshold
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]

    df_filtered = df.drop(columns=drop_cols)

    print(f"  Correlation filter (|r| > {threshold}): dropped {len(drop_cols)} columns")
    if drop_cols:
        for col in drop_cols:
            # Find which column it was most correlated with
            partner = upper[col].idxmax()
            r_value = upper[col].max()
            print(f"    - {col}  (|r|={r_value:.4f} with {partner})")

    return df_filtered, drop_cols


def select_features(df, variance_thresh=0.01, correlation_thresh=0.95):
    """Run the full feature selection pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric feature matrix (no labels, no IPs).
    variance_thresh : float
        Minimum variance to keep a feature.
    correlation_thresh : float
        Maximum allowed |Pearson r| between two features.

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix.
    list[str]
        Names of the selected (kept) columns.
    dict
        Summary with keys 'dropped_variance', 'dropped_correlation',
        'original_count', 'final_count'.
    """
    print("\n" + "=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)

    original_count = len(df.columns)
    print(f"  Starting with {original_count} features")

    # Stage 1: Variance threshold
    df, dropped_var = variance_threshold(df, threshold=variance_thresh)

    # Stage 2: Correlation filter
    df, dropped_corr = correlation_filter(df, threshold=correlation_thresh)

    final_count = len(df.columns)
    selected_columns = df.columns.tolist()

    print(f"\n  Feature selection complete:")
    print(f"    {original_count} → {final_count} features "
          f"(removed {original_count - final_count})")

    summary = {
        "dropped_variance": dropped_var,
        "dropped_correlation": dropped_corr,
        "original_count": original_count,
        "final_count": final_count,
    }

    return df, selected_columns, summary


# ------------------------------------------------------------------
# Standalone verification
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "processed", "friday_clean.csv"
    )
    DATA_PATH = os.path.normpath(DATA_PATH)

    if not os.path.exists(DATA_PATH):
        print(f"Clean dataset not found at {DATA_PATH}")
        print("Run clean_dataset.py first.")
        sys.exit(1)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # Keep only numeric feature columns (drop IPs, Timestamp, labels)
    drop_cols = [
        "Src IP dec", "Dst IP dec", "Timestamp",
        "Label", "Attempted Category",
    ]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Replace inf with NaN then fill with 0 for variance / corr calculation
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    filtered_df, selected, summary = select_features(feature_df)

    print(f"\nSelected features ({len(selected)}):")
    for i, col in enumerate(selected, 1):
        print(f"  {i:>3}. {col}")
    print("\nDone.")

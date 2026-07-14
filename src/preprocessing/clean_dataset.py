import os
import numpy as np
import pandas as pd

# ======================================================
# Configuration
# ======================================================

DATA_PATH = r"/home/soham/Downloads/ISI Project/data/raw/friday.csv"

OUTPUT_PATH = r"/home/soham/Downloads/ISI Project/data/processed/friday_clean.csv"

# ======================================================
# Load Dataset
# ======================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully!")

# ======================================================
# Original Shape
# ======================================================

print("\n" + "=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ======================================================
# Remove Duplicate Rows
# ======================================================

print("\n" + "=" * 60)
print("REMOVING DUPLICATE ROWS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Duplicate Rows Found : {duplicates}")

df = df.drop_duplicates()

print(f"Remaining Rows : {df.shape[0]}")

# ======================================================
# Remove Constant Columns
# ======================================================

print("\n" + "=" * 60)
print("REMOVING CONSTANT COLUMNS")
print("=" * 60)

constant_columns = []

for col in df.columns:

    if df[col].nunique() == 1:

        constant_columns.append(col)

print("Constant Columns Found:")

for col in constant_columns:
    print(" -", col)

df.drop(columns=constant_columns, inplace=True)

print(f"Removed {len(constant_columns)} constant columns.")

# ======================================================
# Missing Values
# ======================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:

    print("No Missing Values Found.")

else:

    print(missing)

    print("\nFilling Missing Values with Median...")

    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:

        df[col].fillna(df[col].median(), inplace=True)

# ======================================================
# Infinite Values
# ======================================================

print("\n" + "=" * 60)
print("INFINITE VALUE CHECK")
print("=" * 60)

numeric_df = df.select_dtypes(include=np.number)

inf_count = np.isinf(numeric_df).sum().sum()

print(f"Infinite Values Found : {inf_count}")

if inf_count > 0:

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    numeric_df = numeric_df.fillna(numeric_df.median())

    df[numeric_df.columns] = numeric_df

    print("Infinite Values Replaced.")

# ======================================================
# Final Verification
# ======================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print(f"Missing Values : {df.isnull().sum().sum()}")

numeric_df = df.select_dtypes(include=np.number)

print(f"Infinite Values : {np.isinf(numeric_df).sum().sum()}")

# ======================================================
# Save Dataset
# ======================================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\nClean Dataset Saved Successfully!")

print(OUTPUT_PATH)
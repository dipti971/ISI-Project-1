import pandas as pd
import numpy as np

# ==========================
# Load Dataset
# ==========================

DATA_PATH = r"C:\Users\soham\Downloads\ISI Project\data\raw\friday.csv"

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully!\n")

# ==========================
# Dataset Shape
# ==========================

print("=" * 60)
print("1. DATASET SHAPE")
print("=" * 60)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ==========================
# Column Names
# ==========================

print("\n" + "=" * 60)
print("2. COLUMN NAMES")
print("=" * 60)

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

# ==========================
# Data Types
# ==========================

print("\n" + "=" * 60)
print("3. DATA TYPES")
print("=" * 60)

print(df.dtypes)

# ==========================
# Missing Values
# ==========================

print("\n" + "=" * 60)
print("4. MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

missing = missing[missing > 0]

if missing.empty:
    print("No Missing Values Found")
else:
    print(missing)

# ==========================
# Duplicate Rows
# ==========================

print("\n" + "=" * 60)
print("5. DUPLICATE ROWS")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Duplicate Rows:", duplicates)

# ==========================
# Infinite Values
# ==========================

print("\n" + "=" * 60)
print("6. INFINITE VALUES")
print("=" * 60)

numeric_df = df.select_dtypes(include=np.number)

inf_count = np.isinf(numeric_df).sum().sum()

print("Infinite Values:", inf_count)

# ==========================
# Label Distribution
# ==========================

print("\n" + "=" * 60)
print("7. LABEL DISTRIBUTION")
print("=" * 60)

print(df["Label"].value_counts())

# ==========================
# Statistical Summary
# ==========================

print("\n" + "=" * 60)
print("8. STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())

# ==========================
# First Five Rows
# ==========================

print("\n" + "=" * 60)
print("9. FIRST FIVE ROWS")
print("=" * 60)

print(df.head())
# ==========================
# Unique Values
# ==========================

print("\n" + "=" * 60)
print("10. UNIQUE VALUES")
print("=" * 60)

for col in df.columns:
    print(f"{col}: {df[col].nunique()}")
# ==========================
# Memory Usage
# ==========================

print("\n" + "=" * 60)
print("11. MEMORY USAGE")
print("=" * 60)

memory = df.memory_usage(deep=True).sum() / 1024**2

print(f"Memory Usage : {memory:.2f} MB")
# ==========================
# Label Percentage
# ==========================

print("\n" + "=" * 60)
print("12. LABEL PERCENTAGE")
print("=" * 60)

print(
    (df["Label"].value_counts(normalize=True) * 100).round(2)
)    
# ==========================
# Constant Columns
# ==========================

print("\n" + "=" * 60)
print("13. CONSTANT COLUMNS")
print("=" * 60)

for col in df.columns:
    if df[col].nunique() == 1:
        print(col)
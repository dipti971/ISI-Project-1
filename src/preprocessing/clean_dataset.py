import pandas as pd

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

print("Dataset Loaded Successfully!")

# ==========================
# Original Dataset Shape
# ==========================

print("\n" + "=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print("Rows   :", df.shape[0])
print("Columns:", df.shape[1])

# ==========================
# Remove Duplicate Rows
# ==========================

print("\nRemoving duplicate rows...")

duplicates = df.duplicated().sum()

print("Duplicate Rows Found:", duplicates)

df = df.drop_duplicates()

print("Duplicate Rows Removed!")

# ==========================
# Remove Constant Columns
# ==========================

print("\nRemoving constant columns...")

constant_columns = []

for col in df.columns:

    if df[col].nunique() == 1:

        constant_columns.append(col)

print("Constant Columns:")

for col in constant_columns:
    print(col)

df = df.drop(columns=constant_columns)

print("Constant Columns Removed!")

# ==========================
# Final Dataset Shape
# ==========================

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)

print("Rows   :", df.shape[0])
print("Columns:", df.shape[1])

# ==========================
# Save Clean Dataset
# ==========================

OUTPUT_PATH = r"C:\Users\soham\Downloads\ISI Project\data\processed\friday_clean.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nClean dataset saved successfully!")

print("Location:")
print(OUTPUT_PATH)
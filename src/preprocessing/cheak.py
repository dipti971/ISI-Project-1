import pandas as pd

df = pd.read_csv(r"C:\Users\soham\Downloads\ISI Project\data\processed\friday_clean.csv")

# Remove only the label column
X = df.drop(columns=["Label"])

print("="*50)
print("Duplicate feature vectors")
print("="*50)  

print(X.duplicated().sum())

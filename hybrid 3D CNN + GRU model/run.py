import pandas as pd

csv_path = "final_valid.csv"
df = pd.read_csv(csv_path)
print(df.columns)
print(df.head())

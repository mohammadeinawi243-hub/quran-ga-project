# explore.py

import pandas as pd

print("1) Reading quran_data.csv file...")
df = pd.read_csv('quran_data.csv')

print("2) Data shape (rows, cols):", df.shape)
print("3) Column names:")
print(df.columns.tolist())

print("4) First 5 rows:")
print(df.head())

print("5) Missing values in each column:")
print(df.isnull().sum())

print("6) Saving clean version to quran_clean.csv (removing missing rows in important columns)...")
df_clean = df.dropna(subset=['revelation_place', 'verses'])
df_clean.to_csv('quran_clean.csv', index=False)
print("Saved -> quran_clean.csv")

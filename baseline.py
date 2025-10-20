# baseline.py

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
from scipy.sparse import hstack, csr_matrix

print("Reloading data and vectorizer ...")
df = pd.read_csv('quran_clean.csv')
tf = joblib.load('tf_vectorizer.joblib')
num_cols = np.load('num_cols.npy', allow_pickle=True).tolist()

X_num = df[num_cols].fillna(0).values
X_text = tf.transform(df['verses'].astype(str).values)
X = hstack([csr_matrix(X_num), X_text])
y = df['revelation_place'].astype(str).values

print("Running Logistic Regression (cross-validation)...")
model = LogisticRegression(max_iter=1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print("Fold scores:", scores)
print("Average accuracy (baseline):", scores.mean())

with open('baseline_result.txt', 'w') as f:
    f.write(str(scores.mean()))

print("Saved -> baseline_result.txt")

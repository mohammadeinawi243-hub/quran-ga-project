# prepare_features.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib
import numpy as np

print("Loading quran_clean.csv ...")
df = pd.read_csv('quran_clean.csv')

# الأعمدة العددية المهمة
print("Preparing numeric features ...")
num_cols = ['verses_count', 'words_count', 'letters_count']
for c in num_cols:
    if c not in df.columns:
        print("Warning:", c, "not found. Creating zeros instead.")
        df[c] = 0

X_num = df[num_cols].fillna(0).values

# تحويل النصوص إلى أرقام (TF-IDF)
print("Transforming verses text to TF-IDF features ...")
texts = df['verses'].astype(str).values
tf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_text = tf.fit_transform(texts)

# دمج الميزات
print("Combining numeric + text features ...")
X_num_sparse = csr_matrix(X_num)
X = hstack([X_num_sparse, X_text])

# العمود الهدف
y = df['revelation_place'].astype(str).values

# حفظ الأدوات المساعدة
print("Saving vectorizer and numeric columns ...")
joblib.dump(tf, 'tf_vectorizer.joblib')
np.save('num_cols.npy', np.array(num_cols))

print("Features prepared successfully!")
print("X shape:", X.shape)
print("Rows:", X.shape[0])
print("Features:", X.shape[1])

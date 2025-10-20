# evaluate_selected.py
# مقارنة بين النموذج الأساسي (baseline) والنموذج بعد تطبيق الخوارزمية الجينية (GA)

import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import GradientBoostingClassifier

print("Loading data and vectorizer ...")  # تحميل البيانات وملفات التحويل
df = pd.read_csv('quran_clean.csv')
tf = joblib.load('tf_vectorizer.joblib')
num_cols = np.load('num_cols.npy', allow_pickle=True).tolist()

# إعداد الميزات
X_num = df[num_cols].fillna(0).values
X_text = tf.transform(df['verses'].astype(str).values)
X = hstack([csr_matrix(X_num), X_text])
y = df['revelation_place'].astype(str).values

# قراءة نتيجة baseline 
try:
    with open('baseline_result.txt', 'r') as f:
        baseline = float(f.read().strip())
except:
    print("baseline_result.txt not found, recalculating baseline...")
    model = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

# تحميل أفضل الميزات التي اختارتها الخوارزمية الجينية
mask = np.load('best_mask.npy')
cols = np.where(mask == 1)[0]
X_sel = X[:, cols]

# تدريب النموذج بعد اختيار الميزات
model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_sel, y, cv=cv, scoring='accuracy')

# طباعة النتائج
print("Baseline accuracy:", baseline)          # دقة النموذج الأساسي
print("Accuracy after GA:", scores.mean())     # دقة النموذج بعد اختيار الميزات
print("Selected features count:", cols.size)   # عدد الميزات التي تم اختيارها

# حفظ النتائج في ملف
with open('comparison.txt', 'w') as f:
    f.write(f"baseline: {baseline}\n")
    f.write(f"after_ga: {scores.mean()}\n")
    f.write(f"selected_count: {cols.size}\n")

print("Saved -> comparison.txt")  # تم حفظ الملف

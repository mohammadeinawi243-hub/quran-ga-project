# ga_feature_selection_improved.py

import numpy as np
import random
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import hstack, csr_matrix
import time
import matplotlib.pyplot as plt

#تثبيت العشوائية 
random.seed(1)
np.random.seed(1)

print("Loading data ...")
df = pd.read_csv('quran_clean.csv')
tf = joblib.load('tf_vectorizer.joblib')
num_cols = np.load('num_cols.npy', allow_pickle=True).tolist()

# تجهيز البيانات
X_num = df[num_cols].fillna(0).values
X_text = tf.transform(df['verses'].astype(str).values)
X = hstack([csr_matrix(X_num), X_text])
y = df['revelation_place'].astype(str).values

n_features = X.shape[1]
print("Total features:", n_features)

# إعدادات الخوارزمية 
POP_SIZE = 40            # عدد الأفراد
GENERATIONS = 25         # عدد الأجيال
P_MUT = 0.01             # احتمال التحور
MIN_FEATURES = 150       # الحد الأدنى للميزات
PENALTY_FACTOR = 0.0003  # عقوبة صغيرة

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# دالة اللي تحسب جودة الكروموسوم
def fitness(chrom):
    s = chrom.sum()
    if s == 0 or s < MIN_FEATURES:
        return -1.0  # نرفض الحلول الصغيرة جدًا
    cols = np.where(chrom == 1)[0]
    X_sel = X[:, cols]
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
    try:
        score = cross_val_score(model, X_sel, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    except Exception as e:
        print("Eval error:", e)
        return -1.0
    penalty = PENALTY_FACTOR * cols.size
    score = score - penalty
    return round(score, 5)

# إنشاء الجيل الأول من السكان
def init_population(pop_size, n):
    pop = []
    for _ in range(pop_size):
        p = 0.1
        chrom = (np.random.rand(n) < p).astype(int)
        if chrom.sum() < MIN_FEATURES:
            ones = np.random.choice(range(n), size=MIN_FEATURES, replace=False)
            chrom[ones] = 1
        pop.append(chrom)
    return pop

def tournament_selection(scored, k=3):
    selected = []
    for _ in range(len(scored)):
        participants = random.sample(scored, k)
        participants.sort(key=lambda x: x[0], reverse=True)
        selected.append(participants[0][1])
    return selected

def crossover(a, b):
    pt = random.randint(1, len(a)-1)
    child = np.concatenate([a[:pt], b[pt:]])
    return child

def mutate(child, p_mut):
    for i in range(len(child)):
        if random.random() < p_mut:
            child[i] = 1 - child[i]
    return child

def run_ga():
    start = time.time()
    pop = init_population(POP_SIZE, n_features)
    best = (-1, None)
    history = []
    for gen in range(GENERATIONS):
        scored = []
        for ind in pop:
            sc = fitness(ind)
            scored.append((sc, ind))
        scored.sort(key=lambda x: x[0], reverse=True)
        gen_best = scored[0]
        history.append(gen_best[0])
        print(f"Gen {gen} best = {gen_best[0]:.4f} selected = {gen_best[1].sum()}")
        # الاحتفاظ بأفضل اثنين (elitism)
        elites = [scored[0][1].copy(), scored[1][1].copy()] if len(scored) > 1 else [scored[0][1].copy()]
        if gen_best[0] > best[0]:
            best = gen_best
        # اختيار الآباء وإنشاء جيل جديد
        parents = tournament_selection(scored, k=3)
        new_pop = elites.copy()
        while len(new_pop) < POP_SIZE:
            a = random.choice(parents)
            b = random.choice(parents)
            child = crossover(a, b)
            child = mutate(child, P_MUT)
            if child.sum() < MIN_FEATURES:
                ones = np.random.choice(range(n_features), size=MIN_FEATURES, replace=False)
                child[ones] = 1
            new_pop.append(child)
        pop = new_pop
    elapsed = time.time() - start
    print("Done. Time:", round(elapsed, 2), "seconds")
    return best, history

best, history = run_ga()

print("\nBest fitness:", best[0], "Selected features:", best[1].sum())
np.save('best_mask.npy', best[1])
print("Saved best_mask.npy")

# رسم تطور الدقة عبر الأجيال
plt.plot(history)
plt.title("GA Fitness Progress")
plt.xlabel("Generation")
plt.ylabel("Fitness (Accuracy)")
plt.show()

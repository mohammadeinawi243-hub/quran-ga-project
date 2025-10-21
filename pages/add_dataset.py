# pages/add_dataset.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Add New Dataset", page_icon="➕", layout="wide")

st.title("➕ Add New Dataset for Analysis")
st.write("Upload a new CSV file and run Baseline + GA analysis.")

uploaded = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ File uploaded successfully!")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head())

    if 'revelation_place' in df.columns and 'verses' in df.columns:
        y = df['revelation_place'].astype(str).values
        tf = TfidfVectorizer(max_features=500)
        X_text = tf.fit_transform(df['verses'].astype(str).values)
        X = X_text

        st.info("Calculating baseline accuracy...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=1000)
        base_score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
        st.write(f"**Baseline Accuracy:** {base_score:.4f}")

        if st.button("🚀 Run GA Analysis"):
            st.info("Running Genetic Algorithm... (this may take some time)")
            progress = st.progress(0)
            n = X.shape[1]
            pop_size, gens, p_mut = 8, 4, 0.02

            def fitness(chrom):
                if chrom.sum() == 0:
                    return 0
                cols = np.where(chrom == 1)[0]
                X_sel = X[:, cols]
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                score = cross_val_score(model, X_sel, y, cv=cv, scoring='accuracy').mean()
                return score - 0.0005 * cols.size

            pop = [np.random.choice([0, 1], size=n, p=[0.98, 0.02]).astype(int) for _ in range(pop_size)]
            best = (0, None)

            for g in range(gens):
                scored = [(fitness(ind), ind) for ind in pop]
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored[0][0] > best[0]:
                    best = scored[0]
                progress.progress((g + 1) / gens)
                new_pop = []
                parents = [ind for _, ind in scored[:max(2, pop_size // 2)]]
                while len(new_pop) < pop_size:
                    a, b = random.sample(parents, 2)
                    pt = random.randint(1, n - 1)
                    child = np.concatenate([a[:pt], b[pt:]])
                    for i in range(n):
                        if random.random() < p_mut:
                            child[i] = 1 - child[i]
                    new_pop.append(child)
                pop = new_pop

            ga_score = best[0]
            selected = best[1].sum()
            st.success(f"GA finished! Accuracy: {ga_score:.4f}, Selected features: {selected}")

            plt.figure(figsize=(4, 4))
            plt.bar(["Baseline", "After GA"], [base_score * 100, ga_score * 100],
                    color=["#FF4B4B", "#2ECC71"])
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy Comparison")
            st.pyplot(plt)

    else:
        st.warning("The uploaded file must contain 'revelation_place' and 'verses' columns.")
else:
    st.info("Please upload a CSV file to start.")

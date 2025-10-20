# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, hstack
import time, random

st.set_page_config(page_title="Quran GA Project", page_icon="📘", layout="centered")
st.title("📘 Quran GA Project - Genetic Algorithm (Student Version)")
st.write("This web app allows you to upload a CSV file and compare results using GA and baseline models 🌱")

# ========================================
# Step 1 - Upload CSV file
# ========================================
uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded is not None:
    st.success("✅ File uploaded successfully!")
    df = pd.read_csv(uploaded)
    st.write("### Preview of uploaded file:")
    st.dataframe(df.head())

    # check if necessary columns exist
    if "revelation_place" in df.columns and "verses" in df.columns:
        # drop missing rows
        df = df.dropna(subset=["revelation_place", "verses"])
        y = df["revelation_place"].astype(str).values
        num_cols = [c for c in df.columns if df[c].dtype != 'object' and c != 'revelation_place']
        X_num = df[num_cols].fillna(0).values

        # convert text to simple numeric representation (word counts)
        from sklearn.feature_extraction.text import CountVectorizer
        tf = CountVectorizer(max_features=500)
        X_text = tf.fit_transform(df["verses"].astype(str))
        X = hstack([csr_matrix(X_num), X_text])

        # ========================================
        # Step 2 - Baseline accuracy
        # ========================================
        st.info("Calculating baseline model accuracy (please wait)...")
        model = LogisticRegression(max_iter=1000)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        base_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        st.write(f"**Baseline Accuracy:** {base_acc:.4f}")

        # ========================================
        # Step 3 - Simple GA
        # ========================================
        st.info("Running Genetic Algorithm... (takes ~30-60 sec)")

        n_features = X.shape[1]
        POP_SIZE = 10
        GENERATIONS = 5
        P_MUT = 0.02

        random.seed(1)
        np.random.seed(1)

        def fitness(chrom):
            if chrom.sum() == 0:
                return 0.0
            cols = np.where(chrom == 1)[0]
            X_sel = X[:, cols]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            try:
                score = cross_val_score(model, X_sel, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()
            except:
                score = 0.0
            penalty = 0.0005 * cols.size
            return score - penalty

        def run_ga():
            pop = [np.random.choice([0,1], size=n_features, p=[0.95,0.05]) for _ in range(POP_SIZE)]
            best = (0, None)
            for g in range(GENERATIONS):
                st.write(f"➡️ Generation {g+1}/{GENERATIONS}")
                scored = [(fitness(ind), ind) for ind in pop]
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored[0][0] > best[0]:
                    best = scored[0]
                parents = [ind for _, ind in scored[:POP_SIZE//2]]
                new_pop = parents.copy()
                while len(new_pop) < POP_SIZE:
                    a, b = random.sample(parents, 2)
                    pt = random.randint(1, n_features-1)
                    child = np.concatenate([a[:pt], b[pt:]])
                    for i in range(n_features):
                        if random.random() < P_MUT:
                            child[i] = 1 - child[i]
                    new_pop.append(child)
                pop = new_pop
            return best

        start_time = time.time()
        best = run_ga()
        ga_acc = best[0]
        selected = best[1].sum()
        duration = time.time() - start_time

        st.success(f"✅ GA Finished in {duration:.1f} sec")
        st.write(f"**GA Accuracy:** {ga_acc:.4f}")
        st.write(f"**Selected Features Count:** {selected}")

        # ========================================
        # Step 4 - Compare visually
        # ========================================
        st.subheader("📊 Comparison Chart")
        st.bar_chart({
            "Baseline": [base_acc * 100],
            "After GA": [ga_acc * 100]
        })

        # ========================================
        # Step 5 - Interpretation
        # ========================================
        if ga_acc > base_acc:
            st.success("🎉 GA improved the accuracy! Great job.")
        else:
            st.warning("⚠️ GA did not improve accuracy this time.")

    else:
        st.error("❌ The file must contain columns: 'revelation_place' and 'verses'.")
else:
    st.info("👆 Please upload your CSV file to begin.")

st.markdown("---")
st.caption("Made by Mohammad Einawi ")

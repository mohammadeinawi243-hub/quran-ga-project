import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import random
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Quran GA Project", page_icon="📘", layout="wide")

st.title("📘 Quran GA Project - Full Online Analyzer (Student Version)")
st.write("Upload any CSV with columns `revelation_place` and `verses` (or a numeric dataset). App will run baseline and a simple GA feature selection.")

# -------------------------
# Helper: load vectorizer & numeric columns if available
# -------------------------
def load_setup():
    tf = None
    num_cols = []
    have_tf = os.path.exists("tf_vectorizer.joblib")
    have_num = os.path.exists("num_cols.npy")
    try:
        if have_tf:
            tf = joblib.load("tf_vectorizer.joblib")
    except Exception as e:
        tf = None
    try:
        if have_num:
            num_cols = np.load("num_cols.npy", allow_pickle=True).tolist()
    except Exception as e:
        num_cols = []
    return tf, num_cols

# -------------------------
# Helper: prepare features from DataFrame
# returns X (sparse), y (array), feature_names list
# -------------------------
def prepare_features(df, tf=None, num_cols=None, tf_max_features=1000):
    # Make sure required columns exist
    # If it's text dataset with 'verses' and 'revelation_place'
    if 'revelation_place' in df.columns and 'verses' in df.columns:
        df = df.dropna(subset=['revelation_place', 'verses'])
        y = df['revelation_place'].astype(str).values

        # numeric columns: try to use provided num_cols or infer numeric columns
        if not num_cols:
            inferred_num = [c for c in df.columns if df[c].dtype != 'object' and c != 'revelation_place']
        else:
            inferred_num = [c for c in num_cols if c in df.columns]

        X_num = df[inferred_num].fillna(0).values if inferred_num else np.zeros((len(df), 0))

        # text features: if a tf vectorizer provided use it, else create a TF-IDF with limited features
        if tf is None:
            tf = TfidfVectorizer(max_features=min(tf_max_features, 1000))
            X_text = tf.fit_transform(df['verses'].astype(str).values)
            tf_is_new = True
        else:
            X_text = tf.transform(df['verses'].astype(str).values)
            tf_is_new = False

        # combine numeric and text
        if X_num.shape[1] > 0:
            X = hstack([csr_matrix(X_num), X_text])
            feature_names = inferred_num + [f"tf_{i}" for i in range(X_text.shape[1])]
        else:
            X = X_text
            feature_names = [f"tf_{i}" for i in range(X_text.shape[1])]

        return X, y, feature_names, tf, inferred_num, tf_is_new
    else:
        # Fallback: dataset without text. Use numeric columns only.
        y = None
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            return None, None, [], None, [], False
        X = csr_matrix(numeric.values)
        feature_names = numeric.columns.tolist()
        # Note: no y here, user must choose target later
        return X, None, feature_names, None, feature_names, False

# -------------------------
# Fitness for GA
# -------------------------
def fitness(chrom, X, y, cv):
    if chrom.sum() == 0:
        return 0.0
    cols = np.where(chrom == 1)[0]
    X_sel = X[:, cols]
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    try:
        score = cross_val_score(model, X_sel, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    except Exception:
        score = 0.0
    penalty = 0.0005 * cols.size
    return score - penalty

# -------------------------
# Simple GA runner (keeps it small by default)
# returns best_score, best_mask
# -------------------------
def run_ga(X, y, pop_size=10, generations=6, p_mut=0.02, min_features=10, progress_callback=None):
    n = X.shape[1]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pop = [np.random.choice([0,1], size=n, p=[0.98,0.02]).astype(int) for _ in range(pop_size)]
    # ensure min features
    for i in range(len(pop)):
        if pop[i].sum() < min_features:
            ones = np.random.choice(range(n), size=min_features, replace=False)
            pop[i][ones] = 1
    best = (0.0, pop[0].copy())
    for gen in range(generations):
        scored = []
        for ind in pop:
            sc = fitness(ind, X, y, cv)
            scored.append((sc, ind))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best[0]:
            best = (scored[0][0], scored[0][1].copy())
        # selection: top half
        parents = [ind for _, ind in scored[:max(2, pop_size//2)]]
        new_pop = parents.copy()
        while len(new_pop) < pop_size:
            a, b = random.sample(parents, 2)
            pt = random.randint(1, n-1)
            child = np.concatenate([a[:pt], b[pt:]])
            # mutation
            for i in range(n):
                if random.random() < p_mut:
                    child[i] = 1 - child[i]
            # ensure min_features
            if child.sum() < min_features:
                ones = np.random.choice(range(n), size=min_features, replace=False)
                child[ones] = 1
            new_pop.append(child)
        pop = new_pop
        if progress_callback:
            progress_callback((gen+1)/generations)
    return best

# -------------------------
# Helper: read saved comparison.txt
# -------------------------
def read_results():
    if os.path.exists("comparison.txt"):
        try:
            with open("comparison.txt", "r") as f:
                lines = f.readlines()
            baseline = float(lines[0].split(":")[1].strip())
            ga = float(lines[1].split(":")[1].strip())
            features_cnt = int(lines[2].split(":")[1].strip())
            return baseline, ga, features_cnt
        except Exception as e:
            st.warning(f"Could not read comparison.txt: {e}")
            return None, None, None
    else:
        return None, None, None

# -------------------------
# UI: left column for upload & options, right column for results
# -------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("Upload / Settings")
    uploaded = st.file_uploader("📂 Add File (CSV)", type=["csv"])
    use_default = st.checkbox("Use default quran_data.csv (if no upload)", value=True)
    st.markdown("**GA Settings (keep small for hosted runs):**")
    pop_size = st.number_input("Population size", min_value=4, max_value=60, value=10, step=1)
    generations = st.number_input("Generations", min_value=1, max_value=20, value=6, step=1)
    p_mut = st.slider("Mutation probability", 0.0, 0.1, 0.02, 0.01)
    min_features = st.number_input("Min selected features", min_value=1, max_value=1000, value=10, step=1)
    run_ga_button = st.button("Run GA on uploaded file (may take time)")

with col_right:
    st.header("Dataset & Results")

# Load pre-saved vectorizer & numeric cols if exist
tf, saved_num_cols = load_setup()

# Decide which dataframe to use
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
elif use_default and os.path.exists("quran_data.csv"):
    df = pd.read_csv("quran_data.csv")
else:
    st.info("Upload a CSV or enable default quran_data.csv")
    st.stop()

# quick preview
st.subheader("Data Preview")
st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
st.dataframe(df.head())

# If text dataset (verses + revelation_place)
if 'revelation_place' in df.columns and 'verses' in df.columns:
    st.info("Detected text dataset with 'verses' and 'revelation_place' columns.")
    # prepare features
    with st.spinner("Preparing features (TF-IDF + numeric)..."):
        X, y, feature_names, tf_used, num_cols_used, tf_is_new = prepare_features(df, tf=tf, num_cols=saved_num_cols, tf_max_features=1000)
    if X is None or y is None:
        st.error("Could not prepare features. Check your dataset.")
        st.stop()

    # Baseline (LogisticRegression)
    with st.spinner("Calculating baseline accuracy..."):
        try:
            cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            baseline_model = LogisticRegression(max_iter=1000)
            baseline_score = cross_val_score(baseline_model, X, y, cv=cv, scoring='accuracy').mean()
        except Exception as e:
            st.error(f"Baseline calculation failed: {e}")
            baseline_score = 0.0

    st.subheader("Baseline Result")
    st.write(f"Baseline Accuracy: **{baseline_score:.4f}**")

    # If user presses run GA button
    if run_ga_button:
        st.info("Starting GA... please be patient.")
        progress_bar = st.progress(0.0)
        start = time.time()
        # progress callback to update bar
        def progress_cb(frac):
            try:
                progress_bar.progress(min(max(frac, 0.0), 1.0))
            except:
                pass

        best_score, best_mask = run_ga(X, y, pop_size=pop_size, generations=generations, p_mut=p_mut, min_features=min_features, progress_callback=progress_cb)
        duration = time.time() - start
        if best_mask is None:
            st.error("GA did not find a solution.")
        else:
            selected_count = int(best_mask.sum())
            st.success(f"GA finished in {duration:.1f} sec. GA accuracy: {best_score:.4f}, selected features: {selected_count}")

            # Save mask (optional)
            try:
                np.save("best_mask.npy", best_mask)
            except:
                pass

            # show selected feature names (limit first 200)
            selected_names = [feature_names[i] for i in np.where(best_mask==1)[0]]
            st.write("Selected feature names (first 200):")
            st.write(selected_names[:200])

            # Comparison chart
            fig, ax = plt.subplots()
            ax.bar(["Baseline", "After GA"], [baseline_score*100, best_score*100], color=["#FF4B4B", "#2ECC71"])
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Baseline vs GA")
            st.pyplot(fig)

            # write comparison.txt so professor can see raw numbers if he downloads
            try:
                with open("comparison.txt", "w") as f:
                    f.write(f"baseline: {baseline_score}\n")
                    f.write(f"after_ga: {best_score}\n")
                    f.write(f"selected_count: {selected_count}\n")
                st.success("comparison.txt saved on the server.")
            except Exception as e:
                st.warning(f"Could not save comparison.txt: {e}")

    else:
        # show existing comparison if exists
        baseline_val, ga_val, features_cnt = read_results()
        if ga_val is not None:
            st.subheader("Saved Comparison (existing)")
            st.write(f"Baseline: {baseline_val:.4f}  |  After GA: {ga_val:.4f}  | Selected: {features_cnt}")
            try:
                st.bar_chart({"Baseline":[baseline_val*100], "After GA":[ga_val*100]})
            except:
                pass

else:
    # Non-text dataset: let user pick target column from numeric columns
    st.info("Detected a non-text dataset. You must select a numeric target column for classification.")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        st.error("No numeric columns found, cannot build model.")
        st.stop()
    st.write("Numeric columns found:", list(numeric.columns))
    target = st.selectbox("Choose target column (label)", options=numeric.columns.tolist())
    if target:
        X = numeric.drop(columns=[target]).values
        y = df[target].values
        X = csr_matrix(X)
        # baseline
        with st.spinner("Calculating baseline..."):
            try:
                cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                baseline_score = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv, scoring='accuracy').mean()
            except Exception as e:
                st.error(f"Baseline error: {e}")
                baseline_score = 0.0
        st.write(f"Baseline accuracy: {baseline_score:.4f}")

        if st.button("Run GA (simple)"):
            st.info("Running GA on numeric data...")
            start = time.time()
            best_score, best_mask = run_ga(X, y, pop_size=pop_size, generations=generations, p_mut=p_mut, min_features=min_features)
            duration = time.time() - start
            st.success(f"GA finished in {duration:.1f} sec. GA accuracy: {best_score:.4f}")
            st.write("Selected count:", int(best_mask.sum()))

# Footer
st.markdown("---")
st.caption("Made by Mohammad Einawi")

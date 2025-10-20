# app.py

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ---------- Helper Function ----------
def read_results():
    """Reads the comparison results from the text file."""
    baseline = "N/A"
    ga_acc = "N/A"
    features = "N/A"

    if os.path.exists("comparison.txt"):
        with open("comparison.txt", "r") as f:
            for line in f:
                if "baseline" in line:
                    baseline = line.split(":")[1].strip()
                elif "after_ga" in line:
                    ga_acc = line.split(":")[1].strip()
                elif "selected_count" in line:
                    features = line.split(":")[1].strip()
    return baseline, ga_acc, features

# ---------- Page Configuration ----------
st.set_page_config(page_title="Quran GA Project", page_icon="📘", layout="wide")

# ---------- Header ----------
col1, col2 = st.columns([6, 1])
with col1:
    st.title("📘 Quran GA Project (Student Version)")
with col2:
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #2C6BED;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 40px;
            width: 120px;
            margin-top: 25px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- Default Data Section ----------
st.write("This is a simple web app made by student 🌱")
st.markdown("---")

# زر تحميل ملف جديد
uploaded_file = st.file_uploader("📂 Add File (CSV)", type=["csv"])

# ---------- Load Default File or Uploaded ----------
if uploaded_file is not None:
    # إذا رفع المستخدم ملف جديد
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    # إذا لم يرفع ملف، نستخدم ملف القرآن الافتراضي
    if os.path.exists("quran_data.csv"):
        df = pd.read_csv("quran_data.csv")
    else:
        st.error("⚠️ Quran data file (quran_data.csv) not found!")
        st.stop()

# ---------- Data Overview ----------
st.subheader("📊 Data Overview")
st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
st.dataframe(df.head())

# ---------- Results Section ----------
st.markdown("### ⚙️ Model Performance Comparison")

baseline, ga, features = read_results()

st.write(f"**Baseline Accuracy:** {baseline}")
st.write(f"**GA Accuracy:** {ga}")
st.write(f"**Selected Features Count:** {features}")

# ---------- Plot ----------
try:
    base_val = float(baseline)
    ga_val = float(ga)
    plt.figure(figsize=(4, 4))
    plt.bar(["Baseline", "After GA"], [base_val * 100, ga_val * 100], color=["#FF4B4B", "#2ECC71"])
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    st.pyplot(plt)
except:
    st.warning("Could not generate chart. Please check data format.")

st.markdown("---")
st.write("**Author:** Mohammad Einawi")
st.write("**GitHub:** [mohammadeinawi243-hub](https://github.com/mohammadeinawi243-hub/quran-ga-project)")
st.write("_ Python Project using Genetic Algorithm and Streamlit_")

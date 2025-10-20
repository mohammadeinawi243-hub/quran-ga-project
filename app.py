# app.py

import streamlit as st
import os

# نحاول قراءة النتائج من ملف المقارنة
def read_results():
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

# عنوان الصفحة
st.set_page_config(page_title="Quran GA Project", page_icon="📘", layout="centered")

# العنوان الرئيسي
st.title("Quran GA Project (Student Version)")
st.write("This is a simple web app created by a beginner student 🌱")

# قراءة النتائج
baseline, ga, features = read_results()

# عرض النتائج
st.subheader("📊 Results Comparison")
st.write(f"**Baseline Accuracy:** {baseline}")
st.write(f"**GA Accuracy:** {ga}")
st.write(f"**Selected Features Count:** {features}")

# تحليل بسيط
if baseline != "N/A" and ga != "N/A":
    try:
        base_val = float(baseline)
        ga_val = float(ga)
        if ga_val > base_val:
            st.success("✅ GA improved the accuracy! Great result.")
        else:
            st.warning("⚠️ GA did not improve accuracy this time.")
    except:
        pass

st.markdown("---")
st.write("**Author:** Mohammad Einawi")
st.write("**GitHub:** [mohammadeinawi243-hub](https://github.com/mohammadeinawi243-hub/quran-ga-project)")
st.write("_Beginner Python Project using Genetic Algorithm and Streamlit_")
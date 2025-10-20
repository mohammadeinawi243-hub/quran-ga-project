# app.py

import streamlit as st
import os
import matplotlib.pyplot as plt

# دالة بسيطة لقراءة النتائج من ملف المقارنة
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


# إعداد الصفحة
st.set_page_config(page_title="Quran GA Project", page_icon="📘", layout="centered")

# العنوان الرئيسي
st.title("Quran Data Project - Genetic Algorithm")
st.write("This web app was created by a student 🌱 using Streamlit and Python.")

# نقرأ النتائج من الملف
baseline, ga, features = read_results()

# نعرض القيم
st.subheader("📊 Results Comparison")
st.write(f"**Baseline Accuracy:** {baseline}")
st.write(f"**GA Accuracy:** {ga}")
st.write(f"**Selected Features Count:** {features}")

# رسم بياني بسيط
if baseline != "N/A" and ga != "N/A":
    try:
        base_val = float(baseline)
        ga_val = float(ga)

        # نصنع رسم بياني باستخدام matplotlib
        labels = ['Baseline', 'After GA']
        values = [base_val * 100, ga_val * 100]

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=['red', 'green'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Comparison (Baseline vs GA)')
        ax.set_ylim(0, 100)

        # عرض الرسم داخل الصفحة
        st.pyplot(fig)

        # تحليل بسيط
        if ga_val > base_val:
            st.success("✅ GA improved the accuracy! Great result.")
        else:
            st.warning("⚠️ GA did not improve accuracy this time.")
    except:
        st.error("Error converting accuracy values.")

st.markdown("---")
st.write("**Author:** Mohammad Einawi")
st.write("**GitHub:** [mohammadeinawi243-hub](https://github.com/mohammadeinawi243-hub/quran-ga-project)")
st.write("_Simple project using Genetic Algorithm and Streamlit_")

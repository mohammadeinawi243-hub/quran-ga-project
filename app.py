# app.py

import streamlit as st
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# 📘 Function to read results from comparison.txt
# --------------------------------------------------
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


# --------------------------------------------------
# 🖥️ Streamlit page setup
# --------------------------------------------------
st.set_page_config(page_title="Quran GA Project", page_icon="📘", layout="centered")

st.title("Quran GA Project (Student Version)")
st.write("This is a simple web app created by student 🌱")

# Read results from file
baseline, ga, features = read_results()

# --------------------------------------------------
# 📊 Show results on screen
# --------------------------------------------------
st.subheader("📊 Results Comparison")
st.write(f"**Baseline Accuracy:** {baseline}")
st.write(f"**GA Accuracy:** {ga}")
st.write(f"**Selected Features Count:** {features}")

# --------------------------------------------------
# 🎨 Draw the accuracy comparison chart
# --------------------------------------------------
if baseline != "N/A" and ga != "N/A":
    try:
        base_val = float(baseline)
        ga_val = float(ga)

        # Small, clean chart
        fig, ax = plt.subplots(figsize=(2.8, 2.2))
        bars = ax.bar(["Baseline", "After GA"], [base_val * 100, ga_val * 100],
                      color=["#FF6B6B", "#27AE60"], width=0.4, edgecolor="black")

        # 🟢🟥 Add percentage labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,  # الموقع فوق العمود
                    f"{height:.2f}%", ha='center', va='bottom', fontsize=8, color='black')

        ax.set_ylabel("Accuracy (%)", fontsize=8)
        ax.set_title("Accuracy Comparison", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Center the chart
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

        # Compare and show message
        if ga_val > base_val:
            st.success("✅ GA improved the accuracy! Great result.")
        else:
            st.warning("⚠️ GA did not improve accuracy this time.")

    except Exception as e:
        st.warning(f"⚠️ Could not generate chart. Please check comparison.txt\nError: {e}")

else:
    st.info("Please make sure comparison.txt exists and contains valid results.")


# --------------------------------------------------
# 🧑‍💻 Footer info
# --------------------------------------------------
st.markdown("---")
st.write("**GitHub:** [mohammadeinawi243-hub](https://github.com/mohammadeinawi243-hub/quran-ga-project)")
st.write("_Python Project using Genetic Algorithm and Streamlit_")
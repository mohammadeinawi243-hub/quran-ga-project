# gui_result.py
# واجهة بسيطة لعرض نتائج المشروع

import tkinter as tk
from tkinter import messagebox
import os

# دالة لقراءة النتائج من ملف المقارنة
def load_results():
    if not os.path.exists("comparison.txt"):
        messagebox.showerror("Error", "File comparison.txt not found!")
        return "N/A", "N/A", "N/A"

    baseline = "N/A"
    ga_acc = "N/A"
    features = "N/A"

    # نحاول نقرأ الملف
    try:
        with open("comparison.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "baseline" in line:
                    baseline = line.split(":")[1].strip()
                elif "after_ga" in line:
                    ga_acc = line.split(":")[1].strip()
                elif "selected_count" in line:
                    features = line.split(":")[1].strip()
    except Exception as e:
        messagebox.showerror("Error", f"Error reading file: {e}")

    return baseline, ga_acc, features

# إنشاء نافذة
window = tk.Tk()
window.title("Quran GA Project - Result Viewer (Student Version)")
window.geometry("400x250")
window.resizable(False, False)

# نقرأ النتائج
baseline_acc, ga_acc, feature_count = load_results()

# العنوان
title_label = tk.Label(window, text="Quran GA Feature Selection", font=("Arial", 14, "bold"))
title_label.pack(pady=10)

# النصوص
tk.Label(window, text="Baseline Accuracy:", font=("Arial", 12)).pack()
lbl_base = tk.Label(window, text=baseline_acc, font=("Arial", 12, "bold"), fg="blue")
lbl_base.pack()

tk.Label(window, text="GA Accuracy:", font=("Arial", 12)).pack()
lbl_ga = tk.Label(window, text=ga_acc, font=("Arial", 12, "bold"), fg="green")
lbl_ga.pack()

tk.Label(window, text="Selected Features Count:", font=("Arial", 12)).pack()
lbl_feat = tk.Label(window, text=feature_count, font=("Arial", 12, "bold"), fg="purple")
lbl_feat.pack()

# زر خروج
btn_exit = tk.Button(window, text="Close", command=window.destroy, bg="red", fg="white", font=("Arial", 11))
btn_exit.pack(pady=15)

# عرض النافذة
window.mainloop()

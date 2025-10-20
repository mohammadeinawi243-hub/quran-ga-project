# app.py
# واجهة ويب بسيطة لعرض نتائج المقارنة بين baseline و GA

from flask import Flask, render_template_string
import os

app = Flask(__name__)

# HTML template as a string (صفحة HTML بسيطة داخل الكود)
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Quran Data Project - GA Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            text-align: center;
            margin-top: 50px;
        }
        .card {
            display: inline-block;
            background: white;
            padding: 25px 40px;
            border-radius: 15px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        .result {
            font-size: 20px;
            margin-top: 15px;
        }
        .bar-container {
            margin-top: 30px;
            display: flex;
            justify-content: center;
            gap: 40px;
        }
        .bar {
            width: 100px;
            background-color: #3498db;
            border-radius: 5px;
            text-align: center;
            color: white;
            padding-top: 5px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Quran Data Project - Genetic Algorithm</h1>
        <div class="result">
            <p><b>Baseline Accuracy:</b> {{ baseline }}</p>
            <p><b>GA Accuracy:</b> {{ ga_acc }}</p>
            <p><b>Selected Features:</b> {{ features }}</p>
        </div>
        <div class="bar-container">
            <div>
                <div class="bar" style="height: {{ baseline*300 }}px; background-color: #e74c3c;">
                    {{ (baseline*100)|round(2) }}%
                </div>
                <p>Baseline</p>
            </div>
            <div>
                <div class="bar" style="height: {{ ga_acc*300 }}px; background-color: #2ecc71;">
                    {{ (ga_acc*100)|round(2) }}%
                </div>
                <p>After GA</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    # قراءة النتائج من ملف المقارنة
    baseline = 0
    ga_acc = 0
    features = 0

    if os.path.exists("comparison.txt"):
        with open("comparison.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "baseline" in line:
                    baseline = float(line.split(":")[1].strip())
                elif "after_ga" in line:
                    ga_acc = float(line.split(":")[1].strip())
                elif "selected_count" in line:
                    features = int(float(line.split(":")[1].strip()))

    return render_template_string(html_template, baseline=baseline, ga_acc=ga_acc, features=features)

if __name__ == "__main__":
    # تشغيل السيرفر المحلي
    print("Starting Flask app on http://127.0.0.1:5000 ...")  # تشغيل التطبيق
    app.run(debug=True)

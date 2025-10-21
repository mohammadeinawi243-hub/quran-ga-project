# Quran GA Project (Student Project)

Hello 👋  
This is my small project for my university assignment.  
I am a student and this project helped me learn about Python, data, and the Genetic Algorithm.

---

## 📘 Project Idea
The goal of this project is to use **Genetic Algorithm (GA)** to select the best features from a Quran dataset  
to improve the model accuracy when predicting the *revelation place* of each surah (Makkah or Madinah).

---

## 🧩 Steps I Did
1. **Data Understanding and Cleaning**  
   - I used `explore.py` to read and clean the dataset `quran_data.csv`.  
   - It removes missing values and saves a clean file called `quran_clean.csv`.

2. **Feature Preparation**  
   - I used TF-IDF for the verses text and added numeric features (like verses count and words count).  
   - This step is done in `prepare_features.py`.

3. **Genetic Algorithm (GA)**  
   - I wrote a simple GA in `ga_feature_selection.py` to choose useful features.  
   - It uses RandomForest or Logistic Regression to test the accuracy for each generation.

4. **Baseline and Comparison**  
   - `baseline.py` is the normal model without GA.  
   - `evaluate_selected.py` compares baseline accuracy vs GA accuracy.

5. **Result**  
   - Baseline accuracy: about **0.89**  
   - After GA accuracy: about **0.92**  
   (Small improvement 👍)

---

## 🌐 Web Interface (Streamlit)
- I built a simple web interface using **Streamlit** in `app.py`.  
- The web page shows:
  - The baseline and GA accuracy comparison.
  - A small colored chart (green/red bars) to visualize improvement.
  - Information about selected features count.  
- It can also be hosted online using **Render.com** or **Streamlit Cloud**.

---

## 📸 Result Images
| Image | Description |
|--------|--------------|
| 🖼️ 1️⃣ | Baseline result (Logistic Regression accuracy) |
| 🖼️ 2️⃣ | GA running generations (console output) |
| 🖼️ 3️⃣ | Comparison result from `comparison.txt` |
| 🖼️ 4️⃣ | Final Streamlit web app result |

---

## 📊 Result Meaning (in my words)
The GA tried to find which features are really helpful for the model.  
Sometimes it takes time, and not always better, but it was fun to test and see how accuracy changes.

---

## 💻 Tools I Used
- Python 3.13  
- Libraries: pandas, numpy, scikit-learn, joblib, scipy, streamlit  
- Editor: Visual Studio Code  
- Git Bash for upload to GitHub  
- Render.com for hosting

---

## 📁 Project Files
| File | Description |
|------|--------------|
| `explore.py` | Read and clean the Quran dataset |
| `prepare_features.py` | Create TF-IDF and numeric feature files |
| `ga_feature_selection.py` | Main genetic algorithm code |
| `baseline.py` | Normal logistic regression model |
| `evaluate_selected.py` | Compare GA result with baseline |
| `comparison.txt` | Final accuracy comparison result |
| `app.py` | Streamlit web interface |
| `quran_clean.csv` | Clean version of dataset |

---

## 🧠 Notes
- I am still learning Python and this is my **first project ever**.  
- I wrote all comments in Arabic to explain my thinking step by step.  
- Maybe the GA can be improved later with more generations or better mutation logic.  
- I learned a lot about data preprocessing, model evaluation, and GitHub usage.

---

## ✨ Author
**GitHub:** [mohammadeinawi243-hub](https://github.com/mohammadeinawi243-hub/quran-ga-project)

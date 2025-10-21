# Quran GA Project (Student Project)

Hello 👋  
This is my small project for my university assignment.  
I am a student and this project helped me learn about Python, data, and Genetic Algorithm.

---

## 📘 Project Idea
The goal of this project is to use **Genetic Algorithm (GA)** to select the best features from a Quran dataset  
to improve the model accuracy when predicting the *revelation place* of each surah (Makkah or Madinah).

---

## 🧩 Steps I did 
1. **Data Understanding and Cleaning**  
   I used `explore.py` to read and clean the dataset `quran_data.csv`.  
   It removes missing values and saves a clean file called `quran_clean.csv`.

2. **Feature Preparation**  
   I used TF-IDF for the verses text and added numeric features (like verses count and words count).  
   This step is done in `prepare_features.py`.

3. **Genetic Algorithm (GA)**  
   I wrote a simple GA in `ga_feature_selection.py` to choose useful features.  
   It uses RandomForest / Logistic Regression to test the accuracy for each generation.

4. **Baseline and Comparison**  
   `baseline.py` is the normal model without GA.  
   `evaluate_selected.py` compares baseline accuracy vs GA accuracy.

5. **Result**  
   - Baseline accuracy: about **0.89**  
   - After GA accuracy: about **0.92**  
   (small improvement 👍)

---

## 📊 Result Meaning (in my words)
The GA tried to find which features are really helpful for the model.  
Sometimes it takes time, and not always better, but it was fun to test and see how accuracy changes.

---

## 💻 Tools I used
- Python 3.13  
- Libraries: pandas, numpy, scikit-learn, joblib, scipy  
- Editor: Visual Studio Code  
- Git Bash for upload to GitHub

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
| `quran_clean.csv` | Clean version of dataset |

---

## 🧠 Notes
- I am still learning Python and this is my **first project ever**.  
- I tried to comment everything clearly in Arabic so my teacher can understand my steps.  
- Maybe the GA can be improved later with more generations or better mutation logic.  
- I learned a lot from this project about data preprocessing and model evaluation.

---

## ✨ Author
**GitHub:** [mohammadeinawi243-hub](https://github.com/mohammadeinawi243-hub)

# Quran Data Project - Feature Selection using Genetic Algorithm

It uses Python and a Genetic Algorithm (GA) to improve a model that predicts the **revelation place** of Quran chapters.

---

## 📊 Project Structure
| File | Description |
|------|--------------|
| `explore.py` | Reads and cleans the Quran dataset. |
| `prepare_features.py` | Converts text into TF-IDF features and merges numeric ones. |
| `baseline.py` | Runs a simple Logistic Regression model as baseline. |
| `ga_feature_selection.py` | Applies a Genetic Algorithm to select the best features. |
| `evaluate_selected.py` | Compares baseline and GA accuracies. |
| `app.py` | Flask web app to display the results visually. |
| `comparison.txt` | File containing final results. |

---

## ⚙️ Requirements
Run the following commands (after creating a virtual environment):

```bash
pip install pandas scikit-learn flask joblib scipy

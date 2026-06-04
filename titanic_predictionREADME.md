# 🚢 Titanic Survival Predictor — ML + GCP Deployment

![Python](https://img.shields.io/badge/Python-3.9-blue)
![GCP](https://img.shields.io/badge/GCP-Vertex%20AI-orange)
![BigQuery](https://img.shields.io/badge/BigQuery-Data%20Warehouse-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-83.4%25-green)

## 📌 Project Overview
An end-to-end Machine Learning project that predicts Titanic passenger 
survival using a Random Forest classifier, deployed as a live REST API 
on Google Cloud Platform (Vertex AI) with BigQuery integration.

---

## 🎯 Key Results
| Metric | Value |
|---|---|
| Best Model | Random Forest |
| Accuracy | 83.4% |
| Cross-Validation | 5-Fold stable |
| Precision (Died) | 90% recall |
| Deployment | GCP Vertex AI |

---

## 🔍 What This Project Finds
- **Gender**: Females had significantly higher survival rates
- **Class**: 1st class passengers survived more than 2nd and 3rd
- **Age & Fare**: Key continuous variables for prediction
- **Model**: Random Forest outperformed Decision Tree, SVM, KNN, 
  and Logistic Regression

---

## 🛠️ Tech Stack
- **Language**: Python 3.9
- **ML Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Cloud**: Google Cloud Platform (GCP)
- **Services**: Vertex AI, BigQuery, Cloud Storage (GCS)
- **Version Control**: GitHub via BigQuery Studio

---

## 📁 Project Structure


---

## 🚀 ML Pipeline Steps
1. **Data Loading** — Titanic dataset (891 passengers)
2. **EDA** — Survival analysis by gender, class, age, fare
3. **Data Cleaning** — Null handling (median/mode imputation)
4. **Encoding** — Label encoding for sex and embarked
5. **Train/Test Split** — 80% train, 20% test (stratified)
6. **Model Training** — 5 algorithms compared
7. **Pruning** — Decision Tree optimized at depth=3
8. **Evaluation** — Confusion matrix, ROC curve, CV
9. **GCS Upload** — model.joblib → gs://titanic_123/
10. **BigQuery** — Cleaned data + predictions stored
11. **Vertex AI** — Live REST endpoint deployed

---

## ☁️ GCP Architecture---

## 📊 Model Comparison
| Model | Accuracy |
|---|---|
| **Random Forest** ✅ | **83.4%** |
| Decision Tree (Pruned) | ~81% |
| Logistic Regression | ~80% |
| SVM | ~82% |
| KNN | ~78% |

---

## 💡 How to Run
```bash
# Clone the repo
git clone https://github.com/rahultwoapl8130/Rahul-kum...

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn \
            google-cloud-bigquery google-cloud-storage \
            google-cloud-aiplatform pandas-gbq joblib

# Run the pipeline
python3 titanic.py
```

---

## 👤 Author
**Rahul Kumar**
- GitHub: rahultwoapl8130
- Project: titanic-498313 (GCP)

---

## 📄 License
This project is open source and available under the MIT License.

Add complete project README with ML pipeline and GCP deployment docs

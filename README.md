# CreditWise Loan System

A supervised machine learning project that predicts whether a loan application is likely to be approved based on applicant and loan-related information.

The project follows an end-to-end ML workflow — from data exploration and preprocessing through final model training and prediction on previously unlabeled applicants.

📄 **For full methodology, model comparisons, and detailed results, see [REPORT.md](docs/REPORT.md).**

---

## 📌 Overview

The goal is to build a classification system that learns from historical loan applications and predicts approval status for new applicants.

```text
Loan_Approved → No (0) / Yes (1)
```

---

## 📊 Dataset

| | |
|---|---:|
| Total records | 1000 |
| Columns | 20 |
| Labeled records (train/eval) | 950 |
| Unlabeled records (final prediction) | 50 |

The 50 unlabeled records were **not** imputed — they were held out and predicted only after the final model was trained on all 950 labeled records. See the report for the full reasoning.

---

## 🗂️ Project Structure

```text
creditwise-loan-system/
│
├── data/
│   ├── creditwise_raw_data.csv
│   └── final_loan_predictions.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_pipeline.ipynb
│   └── 03_final_model.ipynb
│
├── models/
│   └── creditwise_logistic_regression.pkl
│
├── docs/
│   └── REPORT.md
│
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

| Directory/File     | Purpose                                             |
| ------------------ | ---------------------------------------------------- |
| `data/`            | Dataset and generated prediction results             |
| `notebooks/`       | Machine learning experimentation and final pipeline  |
| `models/`          | Saved trained model                                  |
| `docs/`            | Detailed methodology and results report              |
| `src/`             | Source code for future application development      |
| `requirements.txt` | Python dependencies                                  |

---

## 🧰 Technologies Used

Python · Pandas · NumPy · Scikit-learn · Joblib · Jupyter Notebook · Git

---

## 📦 Installation

```bash
git clone https://github.com/yashpal0502/creditwise-loan-system.git
cd creditwise-loan-system
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
jupyter notebook
```

Run the notebooks in order:

```text
01_data_preprocessing.ipynb → 02_model_pipeline.ipynb → 03_final_model_and_prediction.ipynb
```

The trained model is saved to `models/`; predictions are saved to `data/`.

---

## 🏆 Final Model

**Logistic Regression** (C=100, threshold=0.5) — chosen over Gaussian Naive Bayes for its higher ROC-AUC, accuracy, precision, and fewer false positives. Full comparison in the [report](docs/REPORT.md#final-model-selection).

## 📌 Key Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 90.53% |
| Precision | 90.38% |
| Recall    | 78.33% |
| F1 Score  | 83.93% |
| ROC-AUC   | 96.69% |

Trained on 950 labeled applicants → predicted 50 previously unlabeled applicants: **19 approved, 31 rejected**.

---

## 🔐 Data Considerations

The raw dataset is excluded from version control via `.gitignore`. Predictions and the trained model are stored separately from raw data.

---

## 🚀 Future Improvements

- Prediction API & web interface
- Real-time applicant prediction
- Model monitoring & automated testing
- SHAP-based explainability
- Cloud deployment

---

## 👨‍💻 Project Status

**Machine Learning Phase: ✅ Completed** — preprocessing, EDA, encoding, scaling, feature engineering, model comparison, CV, tuning, feature selection, final model, predictions, and documentation are all done.

**Application Phase:** ⬜ Not started — API, web interface, deployment.

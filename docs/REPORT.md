# CreditWise Loan System — Technical Report

This report documents the full machine learning workflow behind the CreditWise Loan System, including preprocessing decisions, feature engineering, model comparisons, tuning, and final results. For a high-level overview, installation, and quickstart, see the main [README](../README.md).

---

## 🔬 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Handle Missing Feature Values
     ↓
Exploratory Data Analysis
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Baseline Models
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Feature Selection
     ↓
Final Model Selection
     ↓
Train on All Labeled Data
     ↓
Predict Unlabeled Applicants
```

---

## 1️⃣ Data Preprocessing

Missing values in feature columns were handled using:

**Numerical features**
```python
SimpleImputer(strategy="mean")
```

**Categorical features**
```python
SimpleImputer(strategy="most_frequent")
```

The target variable was handled separately and was never imputed — the 50 rows with missing `Loan_Approved` were held out entirely for final prediction rather than being filled in, to avoid introducing artificial labels into training.

---

## 2️⃣ Categorical Encoding

Categorical features were converted into numerical representations using `OneHotEncoder`.

Encoded variables:
- `Employment_Status`
- `Marital_Status`
- `Loan_Purpose`
- `Property_Area`
- `Gender`
- `Employer_Category`

```python
OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
    sparse_output=False
)
```

`drop="first"` avoids redundant dummy variables. `handle_unknown="ignore"` lets the model handle an unseen category during prediction.

---

## 3️⃣ Feature Scaling

Numerical features were standardized using `StandardScaler`, included inside the preprocessing pipeline so scaling stays consistent between training and prediction. The target variable was not scaled.

```text
Original Feature → Missing Value Imputation → StandardScaler → Scaled Feature
```

---

## 4️⃣ Feature Engineering

Three additional features were created to capture nonlinear relationships and reduce the effect of skewed income values:

```python
DTI_Ratio_sq = DTI_Ratio ** 2
Credit_Score_sq = Credit_Score ** 2
Applicant_Income_log = np.log1p(Applicant_Income)
```

---

## 5️⃣ Models Evaluated

Three classification algorithms were initially evaluated: Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes — using Accuracy, Precision, Recall, F1, and ROC-AUC.

### Baseline Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| -------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression  |   86.84% |    81.82% | 75.00% |   78.26% |  91.83% |
| KNN                  |   75.79% |    62.50% | 58.33% |   60.34% |  77.78% |
| Gaussian NB          |   86.32% |    85.42% | 68.33% |   75.93% |  93.87% |

KNN performed considerably worse, particularly in ROC-AUC, so further work focused on Logistic Regression and Naive Bayes.

### Feature-Engineered Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| -------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression  |   90.53% |    87.50% | 81.67% |   84.48% |  95.19% |
| KNN                  |   81.58% |    70.49% | 71.67% |   71.07% |  86.91% |
| Gaussian NB          |   87.89% |    81.36% | 80.00% |   80.67% |  96.13% |

Feature engineering substantially improved Logistic Regression in particular.

---

## 🔄 Cross-Validation

5-fold Stratified Cross-Validation was used to keep class distribution consistent across folds:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

| Model               | Mean Accuracy | Mean ROC-AUC |
| -------------------- | ------------: | -----------: |
| Logistic Regression  |        88.42% |       95.31% |
| Gaussian NB          |        90.53% |       96.83% |

---

## 🎛️ Hyperparameter Tuning

`GridSearchCV` was used, optimizing for ROC-AUC.

**Logistic Regression** — `C` tested over `[0.01, 0.1, 1, 10, 100]` → best `C = 100`, CV ROC-AUC `0.9688`.

**Gaussian Naive Bayes** — `var_smoothing` tuned → best `1e-11`, CV ROC-AUC `0.9653`.

---

## ✂️ Feature Selection

The following features were removed to simplify the model while maintaining or improving performance:

```text
Savings, Collateral_Value, Existing_Loans, Marital_Status, Loan_Term, Age, Dependents
```

---

## Final Model Selection

**Selected Logistic Regression** — 5-fold CV ROC-AUC: mean `0.9712`, std `0.0065`

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 90.53% |
| Precision | 90.38% |
| Recall    | 78.33% |
| F1 Score  | 83.93% |
| ROC-AUC   | 96.69% |

Confusion Matrix:
```text
[[125,  5],
 [ 13, 47]]
```

**Tuned Gaussian Naive Bayes** — 5-fold CV ROC-AUC: mean `0.9653`, std `0.0051`

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 90.00% |
| Precision | 84.75% |
| Recall    | 83.33% |
| F1 Score  | 84.03% |
| ROC-AUC   | 96.32% |

**Decision:** Logistic Regression was selected. It provided higher CV ROC-AUC, accuracy, and precision, fewer false positives, and a simpler, more interpretable linear model — despite Naive Bayes edging it out slightly on recall and F1.

---

## 🔮 Final Prediction

The final model was retrained on all 950 labeled records and saved to:
```text
models/creditwise_logistic_regression.pkl
```

It was then used to predict the 50 applicants with missing `Loan_Approved` values.

| Prediction |  Count |
| ---------- | -----: |
| No         |     31 |
| Yes        |     19 |
| **Total**  | **50** |

Results were saved to `data/final_loan_predictions.csv`.

### Prediction Output Format

| Column                     | Description                    |
| -------------------------- | ------------------------------- |
| `Applicant_ID`             | Applicant identifier            |
| `Loan_Approved_Prediction` | Predicted loan approval status  |
| `Approval_Probability`     | Probability of loan approval    |
| `Rejection_Probability`    | Probability of loan rejection   |

```text
Applicant_ID | Loan_Approved_Prediction | Approval_Probability | Rejection_Probability
-------------|--------------------------|----------------------|-----------------------
A001         | Yes                      | 0.87                 | 0.13
A002         | No                       | 0.24                 | 0.76
```

Decision threshold: `Probability >= 0.5 → Yes`, else `No`.

---

## 💾 Saved Model

The saved pipeline bundles preprocessing and the trained model together, so the same steps apply automatically to new applicant data:

```python
import joblib
model = joblib.load("models/creditwise_logistic_regression.pkl")
```

---

## 🧪 Notebook Breakdown

**`01_data_preprocessing.ipynb`** — dataset inspection, missing-value analysis, EDA, encoding, scaling, feature engineering, baseline model evaluation, cross-validation.

**`02_model_pipeline.ipynb`** — preprocessing pipeline, model pipelines, cross-validation, hyperparameter tuning, feature selection, final model comparison and selection.

**`03_final_model_and_prediction.ipynb`** — loads the complete dataset, separates labeled/unlabeled applicants, applies final feature engineering and selection, trains the final model on all labeled data, saves it, predicts the 50 unlabeled applicants, and exports final predictions.
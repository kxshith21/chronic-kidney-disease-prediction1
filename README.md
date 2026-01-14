# chronic-kidney-disease-prediction1
# Chronic Kidney Disease (CKD) Prediction Using Machine Learning

## Overview

This project focuses on building a **machine learning model to predict Chronic Kidney Disease (CKD)** using a real-world medical dataset. The goal is to assist in early detection by classifying patients as either having CKD or not.

The project involves **data cleaning, preprocessing, model training, evaluation, and saving the trained model** for future predictions.

## Dataset

* **Source:** UCI kaggle (CKD dataset)
* **Number of Records:** 400 (example size)
* **Features:** 24 features including age, blood pressure, blood test results, and categorical medical indicators such as rbc, pc, ba, htn, dm, etc.
* **Target Variable:** `classification` (ckd or notckd)

## Data Cleaning & Preprocessing

* Handled **missing values** using mode imputation for categorical data and mean/median for numerical features.
* Standardized categorical values by removing **extra spaces, tabs, and special characters**.
* Encoded categorical features using **LabelEncoder** and manual mapping for yes/no, normal/abnormal, present/notpresent features.
* Converted relevant columns to numeric types for ML model compatibility.
* Ensured target labels were consistent (`ckd = 1`, `notckd = 0`).

## Machine Learning Models

* **Random Forest Classifier:** Primary model due to ability to handle mixed data types and feature importance extraction.
* **Logistic Regression:** Baseline model for comparison.
* **Gradient Boosting Classifier:** Optional model to compare performance.

### Model Selection Criteria

* Evaluated models using **accuracy, precision, recall, and F1-score**.
* **Recall of the CKD class was prioritized** to minimize false negatives (missing a CKD patient).

## Results

* **Random Forest Classifier Performance:**

  * Accuracy: 0.98
  * Recall (CKD): 0.96
  * F1-score: 0.98
* Confirms **high reliability** for identifying CKD patients.

## Model Saving

* Model saved using **joblib**:

```python
import joblib
joblib.dump(model, 'models/random_forest_ckd.pkl')
```

* Allows future prediction without retraining.

## Sample Prediction

```python
import pandas as pd
import joblib

model = joblib.load('models/random_forest_ckd.pkl')

sample_input = {
    'age': 48, 'bp': 80, 'sg': 1.020, 'al': 1, 'su': 0,
    'rbc': 1, 'pc': 1, 'pcc': 0, 'ba': 1, 'bgr': 121,
    'bu': 36, 'sc': 1.2, 'sod': 137, 'pot': 4.5, 'hemo': 11.3,
    'pcv': 33, 'wc': 7800, 'rc': 4.2, 'htn': 1, 'dm': 1, 'cad': 0,
    'appet': 0, 'pe': 1, 'ane': 1
}

X_sample = pd.DataFrame([sample_input])
prediction = model.predict(X_sample)
print('CKD' if prediction[0]==1 else 'Not CKD')
```

## Tools & Technologies

* **Python**
* **Pandas & NumPy** for data handling
* **Scikit-learn** for machine learning
* **Joblib** for saving the trained model

## Key Learnings

* How to clean messy medical datasets with mixed categorical and numerical data.
* Importance of **recall** for medical diagnosis tasks.
* Working with machine learning pipelines and saving models for production-ready deployment.
* Encoding categorical variables safely and handling missing data.

## Disclaimer

⚠ This project is **for educational purposes only** and **should not be used for actual medical diagnosis**.

## GitHub Structure Suggestion

```
CKD-Prediction-ML/
│
├── data/                # CKD dataset CSV
├── notebooks/           # EDA and model building notebooks
├── src/                 # Preprocessing and training scripts
├── models/              # Saved model files
├── README.md            # Project description
└── requirements.txt     # Required Python packages
```

---

**End of README**

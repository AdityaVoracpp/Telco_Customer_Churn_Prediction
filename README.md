# Customer Churn Prediction using Machine Learning

## Overview

A machine learning project focused on predicting customer churn using classification algorithms. The objective is to identify customers who are likely to discontinue a service and help businesses take proactive retention measures.

The project explores multiple classification models, techniques for handling class imbalance, and evaluation strategies to improve churn detection.

## Objectives

- Build machine learning models to predict customer churn.
- Compare Random Forest and XGBoost classifiers.
- Handle class imbalance using class weighting and SMOTE.
- Optimize classification thresholds using validation data.
- Identify important features influencing customer churn.
- Evaluate models using multiple classification metrics.

## Technologies Used

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Imbalanced Learning:** Imbalanced-learn
- **Model Serialization:** Pickle
- **Environment:** Jupyter Notebook

## Models Implemented

The following models were trained and evaluated:

1. Random Forest
2. Random Forest with Class Weighting
3. XGBoost
4. XGBoost with Class Weighting
5. Random Forest with SMOTE
6. XGBoost with SMOTE
7. Random Forest with SMOTE and Class Weighting
8. XGBoost with SMOTE and Class Weighting

## Experimental Results

Models were compared using validation data with a default classification threshold of 0.5.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 79.41% | 65.44% | 47.49% | 55.04% | 82.81% |
| Random Forest (Balanced) | 77.99% | 57.57% | 64.88% | 61.01% | 82.70% |
| XGBoost | 80.39% | 68.40% | 48.50% | 56.75% | **85.10%** |
| XGBoost (Weighted) | 76.13% | 53.49% | **76.92%** | **63.10%** | 84.64% |
| Random Forest + SMOTE | 78.62% | 62.18% | 49.50% | 55.12% | 82.21% |
| Random Forest + SMOTE + Balanced | 78.53% | 61.92% | 49.50% | 55.02% | 82.22% |
| XGBoost + SMOTE | **80.92%** | 66.80% | 55.85% | 60.84% | 84.68% |
| XGBoost + SMOTE + Weighted | 75.42% | 52.56% | 75.59% | 62.00% | 84.45% |

### Key Observations

- Standard XGBoost achieved the highest ROC-AUC of approximately 85.10%, indicating strong discriminatory ability.
- XGBoost with class weighting achieved the highest F1-score of 63.10% and recall of 76.92%.
- Class weighting substantially improved churn detection by increasing the importance of minority-class samples during training.
- SMOTE improved the F1-score of standard XGBoost from 56.75% to 60.84%.
- Combining SMOTE with class weighting did not provide a further improvement over class weighting alone.

### Model Selection

XGBoost with class weighting emerged as the strongest candidate when prioritizing churn detection and balancing precision with recall.

Standard XGBoost remained the strongest candidate in terms of ROC-AUC.

The final model selection is subject to validation-based threshold optimization and evaluation on an untouched test set.

## Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churners who actually churn.
- **Recall:** Proportion of actual churners successfully identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability of the model to distinguish between churners and non-churners across classification thresholds.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── customer_churn_model.pkl
├── requirements.txt
├── README.md
│
└── data/
    └── dataset.csv
```

## Installation and Usage

1. Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Customer-Churn-Prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `Customer_Churn_Prediction.ipynb` and execute the cells.

### Loading the Trained Model

The serialized model can be loaded using Pickle:

```python
import pickle

with open("customer_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(new_customer_data)
```

The input data must contain the same features expected by the trained preprocessing pipeline.

## Business Applications

Customer churn prediction can help businesses:

- Identify customers at risk of leaving.
- Design targeted customer retention campaigns.
- Improve customer onboarding strategies.
- Understand factors associated with customer attrition.
- Allocate retention resources more effectively.

Feature importance analysis can provide additional insights into the factors contributing to churn predictions.

## Future Improvements

- Hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
- Further feature engineering.
- Experimentation with SMOTENC for mixed categorical and numerical data.
- Further threshold optimization.
- Deeper analysis of feature importance and business implications.
- Deployment through a REST API or interactive application.

---

**Note:** The reported experimental results are validation metrics, not final test-set performance. The final test evaluation should be reported separately after model selection and threshold optimization.

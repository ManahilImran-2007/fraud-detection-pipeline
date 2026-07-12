# Credit Card Fraud Detection Pipeline

## Overview
This project builds a supervised machine learning pipeline to detect fraudulent credit card transactions in a highly imbalanced dataset. It was completed as Project 2 of a Data Analytics internship at DecodeLabs.

## Problem Statement
Fraudulent transactions make up only 0.17% of all transactions in this dataset (492 fraud cases out of 284,807 total). This extreme imbalance means a naive model can achieve 99.8% accuracy while catching zero fraud cases. Accuracy was therefore discarded as an evaluation metric in favor of Precision, Recall, and ROC-AUC.

## Dataset
- Source: [Credit Card Fraud Detection (Kaggle, mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, 31 columns
- Features V1-V28 are PCA-transformed (anonymized) for privacy
- `Time` and `Amount` are the only untransformed features
- `Class` is the target: 0 = legitimate, 1 = fraud

## Approach
1. **Train/test split first** (80/20, stratified) to prevent any data leakage before preprocessing
2. **Pipeline construction** using `imblearn.pipeline.Pipeline` to ensure SMOTE and scaling are only ever fit on training folds:
   - Logistic Regression pipeline: StandardScaler → SMOTE → Classifier
   - Random Forest pipeline: SMOTE → Classifier (no scaling needed for tree-based models)
3. **Model training** on both Logistic Regression and Random Forest
4. **Evaluation** using Precision, Recall, ROC-AUC, and confusion matrices instead of accuracy
5. **Hyperparameter tuning** by comparing Random Forest configurations to reduce overfitting and improve generalization

## Results

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.971 |
| Random Forest (default) | 0.83 | 0.83 | 0.964 |
| Random Forest (tuned) | 0.52 | 0.89 | **0.986** |

**Final model: Tuned Random Forest** (n_estimators=30, max_depth=10)
- Caught 87 of 98 fraud cases in the test set
- 79 false positives out of 56,864 legitimate transactions
- Best overall ROC-AUC, indicating strong separation between fraud and legitimate transactions across all thresholds

## Key Takeaways
- **Data leakage prevention**: Applying SMOTE or scaling before the train/test split contaminates the test set with information it shouldn't have. Using `imblearn` pipelines enforces correct ordering structurally.
- **Metric choice matters**: Accuracy is meaningless on imbalanced data. Precision and Recall reveal a real trade-off between catching fraud and avoiding false alarms.
- **Model trade-offs**: Logistic Regression catches the most fraud but drowns it in false positives. Random Forest offers a more balanced, production-realistic result.

## Tools Used
- Python, Pandas
- Scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib

## Author
Manahil

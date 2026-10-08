# Applied AI in Healthcare

## Project-Maternal Health Risk Classification

This project explores the use of machine learning to classify maternal health risk into three ordinal categories:

 Low Risk, Mid Risk and High Risk

The model uses maternal health measurements such as:

- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Blood Sugar
- Body Temperature
- Heart Rate

The project includes data quality assessment, data cleaning, exploratory analysis, machine-learning model comparison, model evaluation, and SHAP-based model explainability.

The evaluated models are:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was selected as the final prototype model based on its held-out test performance and class-level results.

## Dataset

The project uses the Maternal Health Risk dataset available through the UCI Machine Learning Repository.

Dataset source:

https://archive.ics.uci.edu/

The original dataset contains 1,014 records and seven variables. After data-quality assessment and cleaning, the final modelling dataset contains 380 records.

The final dataset contains:

- 203 Low Risk records
- 76 Mid Risk records
- 101 High Risk records

## Methodology

The project workflow includes:

1. Exploratory data analysis
2. Data quality assessment
3. Duplicate and inconsistent-label investigation
4. Data cleaning
5. Feature and target preparation
6. Stratified train-test splitting
7. Feature scaling where appropriate
8. Model training and comparison
9. Five-fold cross-validation
10. Model evaluation using accuracy and balanced accuracy
11. SHAP-based explainability
12. Final model selection
13. Prototype risk prediction

## Model Performance

The models were evaluated using the held-out test set.

| Model | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Logistic Regression | 64.5% | 53.5% |
| Random Forest | 75.0% | 65.6% |
| XGBoost | 82.9% | 75.6% |

XGBoost achieved the strongest held-out test performance. The Mid Risk category remained the most difficult class to classify.

## Explainability

SHAP (SHapley Additive exPlanations) was used to investigate how individual features contributed to the XGBoost model's predictions across the three maternal health risk categories.

Blood Sugar (BS), Systolic Blood Pressure, and Body Temperature showed the strongest overall contributions. Higher values of these features generally contributed towards High Risk predictions in the relevant SHAP plots.

Age, Diastolic Blood Pressure, and Heart Rate generally showed smaller overall contributions.

The Mid Risk category showed more mixed feature effects, suggesting that predictions for this class may depend on combinations of features rather than a single predictor.

SHAP provided an additional layer of model interpretability by helping explain how the XGBoost model used different maternal health measurements when making predictions.

## **Project Structure**

```text
AppliedAI/
├── data/
│   └── Maternal Health Risk Data Set.csv
├── notebooks/
├── results/
├── src/
├── .gitignore
├── Project.ipynb
├── README.md
└── xgb_final_model.pkl
```

## Limitations

This project is a prototype-level machine-learning study and is not a clinical diagnostic system.

The findings should be interpreted cautiously because:

- the final dataset is relatively small
- the source data may not represent the wider maternal population
- the model has not been externally validated
- the Mid Risk class remains more difficult to classify
- the prototype has not been evaluated in a real clinical setting

The model is therefore intended as a prototype decision-support concept rather than a replacement for clinical judgement.

## Purpose

This project was developed as part of my Applied AI studies to explore how machine-learning methods can be applied to a healthcare classification problem and how model performance and interpretability can be evaluated.
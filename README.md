# Maternal Health Risk Classification

This project explores supervised machine learning for classifying maternal health risk into **Low Risk**, **Mid Risk**, and **High Risk** categories using six health-related input features. The notebook covers data-quality investigation, data cleaning, model training and evaluation, SHAP-based explainability, and a small prediction prototype.

> **Important:** This is an educational prototype, not a diagnostic tool. Its predictions and probability outputs are not clinically validated and must not be used to make healthcare decisions.

## Project objectives

- Explore the Maternal Health Risk dataset and investigate data quality.
- Prepare a modelling dataset while documenting duplicate records and inconsistent risk labels.
- Compare Logistic Regression, Random Forest, and XGBoost classifiers.
- Evaluate the models using a stratified held-out test set and five-fold cross-validation.
- Use SHAP to examine which features contribute most to XGBoost predictions.
- Save and reload the selected model and demonstrate predictions on illustrative examples.

## Dataset

The notebook expects the CSV file at:

```text
data/Maternal Health Risk Data Set.csv
```

The dataset is available through the [UCI Machine Learning Repository: Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk). The UCI record describes data collected from healthcare settings in rural Bangladesh and lists the dataset under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence. Please retain attribution to the dataset source when redistributing or adapting it.

**Dataset citation:** Ahmed, M. (2020). *Maternal Health Risk* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DP5D

### Features and target

| Column | Description |
|---|---|
| `Age` | Maternal age |
| `SystolicBP` | Systolic blood pressure |
| `DiastolicBP` | Diastolic blood pressure |
| `BS` | Blood sugar |
| `BodyTemp` | Body temperature |
| `HeartRate` | Heart rate |
| `RiskLevel` | Target category: low risk, mid risk, or high risk |

## Data preparation

The source file contained **1,014 records and 7 columns**. The notebook found no missing values in its initial checks, but identified duplicate records and inconsistent labels.

The cleaning workflow:

1. Removed exact duplicate records, leaving 452 unique rows.
2. Identified 35 predictor profiles where the same six input values were associated with different `RiskLevel` labels. The 71 unique rows associated with these ambiguous profiles were excluded rather than assigned an unsupported label.
3. Removed one remaining record with an implausible `HeartRate` value of 7.
4. Retained records with `Age` below 18 because they had valid feature and target values; their presence is treated as a dataset limitation.

The final modelling dataset contains **380 records**:

| Risk category | Records | Share |
|---|---:|---:|
| Low Risk | 203 | 53.4% |
| Mid Risk | 76 | 20.0% |
| High Risk | 101 | 26.6% |

The target was encoded as `0 = Low Risk`, `1 = Mid Risk`, and `2 = High Risk`. The data was split into training and test sets using a stratified 80:20 split (`random_state=42`), giving 304 training records and 76 test records.

## Models and evaluation

The notebook compares three classifiers:

- **Logistic Regression** — baseline model.
- **Random Forest** — 100 trees (`n_estimators=100`).
- **XGBoost** — 100 estimators, `max_depth=3`, and `learning_rate=0.1`.

The model-comparison table in the notebook reports the following results. Cross-validation uses balanced accuracy over five folds.

| Model | Test accuracy | Test balanced accuracy | CV mean balanced accuracy | CV standard deviation |
|---|---:|---:|---:|---:|
| Logistic Regression | 65.8% | 54.3% | 59.3% | 2.9 percentage points |
| Random Forest | 75.0% | 65.6% | **69.1%** | 7.2 percentage points |
| XGBoost | **82.9%** | **75.6%** | 68.5% | 9.0 percentage points |

XGBoost had the highest held-out test accuracy and balanced accuracy. Random Forest had the highest mean cross-validation balanced accuracy, so the results do not establish that XGBoost is best under every evaluation method.

### Recall by risk category

The held-out test classification report for the updated, scaled Logistic Regression pipeline gives recalls of 88% for Low Risk, 20% for Mid Risk, and 55% for High Risk. The Random Forest and XGBoost recall values in the notebook's class-level comparison are shown alongside them below:

| Risk category | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Low Risk | 88% | 90% | **95%** |
| Mid Risk | 20% | 27% | **47%** |
| High Risk | 55% | 80% | **85%** |

Mid Risk was the most difficult category for all three models. XGBoost had the highest Mid Risk recall of the three, but at 47% it still missed more than half of the Mid Risk examples in the held-out test set.

## SHAP explainability

SHAP (SHapley Additive exPlanations) was applied to the XGBoost model to examine feature contributions for the test data. The overall mean absolute SHAP values recorded in the notebook ranked the features as follows:

| Feature | Mean absolute SHAP value |
|---|---:|
| `BS` | 0.885 |
| `SystolicBP` | 0.496 |
| `BodyTemp` | 0.386 |
| `Age` | 0.176 |
| `DiastolicBP` | 0.136 |
| `HeartRate` | 0.134 |

These values describe how strongly features contribute to model predictions in this fitted model; they do **not** show that a feature causes a health outcome.

## Prototype

The notebook saves the selected XGBoost model as `xgb_final_model.pkl` with `joblib`, reloads it, and verifies that the reloaded model generates the same predictions as the original model for the held-out test data.

It also includes a `predict_maternal_risk(...)` function that accepts `Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, and `HeartRate`, then returns a predicted risk category and, in the probability-enabled version, model-estimated probabilities for each class. The `.pkl` file is generated when the relevant notebook cell is run; it is not required to be present before the notebook is executed.

The model-estimated probabilities have not been clinically validated or calibrated for clinical interpretation.

## Requirements

The notebook uses Python and the following libraries:

- pandas
- NumPy
- seaborn
- Matplotlib
- scikit-learn
- XGBoost
- SHAP
- joblib
- Jupyter (or the Jupyter support in VS Code)

Install the libraries in your chosen Python environment with:

```bash
python -m pip install pandas numpy seaborn matplotlib scikit-learn xgboost shap joblib jupyter
```

## How to run

1. Clone or download this repository.
2. Ensure the dataset file is located at `data/Maternal Health Risk Data Set.csv` relative to the repository root.
3. Open the repository folder in VS Code or start Jupyter from the repository root.
4. Open `Project.ipynb` and select the Python environment where the required libraries are installed.
5. Run the notebook cells from top to bottom. The notebook creates the cleaned modelling data, trains and evaluates the models, runs SHAP analysis, and demonstrates the prediction prototype.

## Limitations and responsible use

- The dataset shrinks to 380 records after duplicate, inconsistent-label, and implausible-value handling.
- The source data and its risk labels may not represent the wider maternal population or healthcare settings.
- Records with ages below 18 remain in the data and should be considered when interpreting model behaviour.
- Mid Risk recall remains low, including 47% for XGBoost on the held-out test set.
- The model has not been externally validated, and its probability outputs are not clinically validated probabilities.
- The prototype is for educational demonstration only and is not a substitute for clinical judgement or professional care.

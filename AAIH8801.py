# %%
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import xgboost
import shap


print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)
print("SHAP:", shap.__version__)


# %%
df = pd.read_csv("data/Maternal Health Risk Data Set.csv")

df.head()

# %%
print("Dataset shape:", df.shape)

# %%
df.info()

# %%
df["RiskLevel"].value_counts()

# %%
df.describe()

# %%
df[df["HeartRate"] < 40]

# %%
df[df.duplicated(keep=False)]

# %%
# Number of completely duplicated rows excluding the first occurance
df.duplicated().sum()

# %%
# Count the occurrences of each unique row and display the 20 most frequent rows

dup_counts = df.value_counts()

dup_counts.head(20)

# %%
print("Heart rate < 40:")
print(df[df["HeartRate"] < 40])

print("\nAge < 18:")
print(df[df["Age"] < 18])
print("\nBlood sugar values:")
print(df["BS"].describe())

print("\nBody temperature values:")
print(df["BodyTemp"].value_counts().sort_index())

# %%
df.groupby("RiskLevel")[[
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]].mean()

# %%
# Plot the distribution of records across maternal health risk levels

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="RiskLevel")

plt.title("Distribution of Maternal Health Risk Levels")
plt.xlabel("Risk Level")
plt.ylabel("Number of Records")

plt.show()

# %%

# Visualise the number of records in each maternal health risk category

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="RiskLevel", y="BS")

plt.title("Blood Sugar Distribution by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Blood Sugar (BS)")

plt.show()

# %%
# Compare blood sugar (BS) distributions across maternal health risk levels
risk_order = ["low risk", "mid risk", "high risk"]

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="RiskLevel",
    y="BS",
    order=risk_order
)

plt.title("Blood Sugar Distribution by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Blood Sugar (BS)")

plt.show()

# %%
# Compare systolic blood pressure distributions across maternal health risk levels

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="RiskLevel",
    y="SystolicBP",
    order=risk_order
)

plt.title("Systolic Blood Pressure by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Systolic Blood Pressure")

plt.show()

# %%
# Compare diastolic blood pressure distributions across maternal health risk levels

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="RiskLevel",
    y="DiastolicBP",
    order=risk_order
)

plt.title("Diastolic Blood Pressure by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Diastolic Blood Pressure")

plt.show()


# %%
risk_order = ["low risk", "mid risk", "high risk"]

# %%
print(risk_order)

# %%


# Compare body temperature distributions across maternal health risk levels

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="RiskLevel",
    y="BodyTemp",
    order=risk_order
)

plt.title("Body Temperature by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Body Temperature")

plt.show()

# %%

# Compare heart rate distributions across maternal health risk levels


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="RiskLevel",
    y="HeartRate",
    order=risk_order
)

plt.title("Heart Rate by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Heart Rate")

plt.show()

# %%

# Compare maternal age distributions across maternal health risk levels

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="RiskLevel",
    y="Age",
    order=risk_order
)

plt.title("Age by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Age")

plt.show()

# %%
# Calculate and display the correlation matrix for the numerical predictor variables


numeric_cols = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]

correlation_matrix = df[numeric_cols].corr()

print(correlation_matrix)

# %%

# Visualise correlations between numerical features using a heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Numerical Features")
plt.show()

# %%
print("Number of duplicate rows:", df.duplicated().sum())

# %%
# Display the first 20 rows that are part of exact duplicate groups, sorted by age and blood pressure
df[df.duplicated(keep=False)].sort_values(
    by=["Age", "SystolicBP", "DiastolicBP"]
).head(20)

# %%
print("Total rows:", len(df))
print("Unique rows:", df.drop_duplicates().shape[0])
print("Duplicate rows:", df.duplicated().sum())

# %% [markdown]
# ## Data Quality Summary
#
# The dataset contains 1,014 records and 7 variables. No missing values were identified during the initial data-quality checks. However, 562 rows were identified as duplicate occurrences, leaving 452 unique records.
#
# The target variable, `RiskLevel`, contains three ordered categories: Low Risk, Mid Risk, and High Risk. Therefore, `RiskLevel` is treated as an ordinal categorical target rather than a nominal categorical variable.
#
# Potentially unusual values identified during exploratory data analysis (EDA) will be investigated before preprocessing. The original dataset will be preserved, and data-cleaning decisions will be applied only to a separate modelling dataset.

# %%
# Check duplicate records by risk level
duplicates = df[df.duplicated(keep=False)]

duplicates["RiskLevel"].value_counts()

# %%
# Check how frequently duplicate records occur
duplicate_counts = df.value_counts()

duplicate_counts.value_counts().sort_index()

# %%
# Check whether identical predictor values have different risk labels

features = df.drop(columns=["RiskLevel"])

label_consistency = (
    df.groupby(list(features.columns))["RiskLevel"]
      .nunique()
)

label_consistency.value_counts()

# %%
# Find predictor profiles that have more than one RiskLevel
inconsistent_profiles = (
    df.groupby(list(features.columns))["RiskLevel"]
      .nunique()
)

inconsistent_profiles = inconsistent_profiles[
    inconsistent_profiles > 1
]

inconsistent_profiles

# %%
# Count how many predictor profiles have 2 or 3 different risk labels

inconsistent_summary = inconsistent_profiles.value_counts().sort_index()

print("Predictor profiles with 2 different RiskLevels:",
      inconsistent_summary.get(2, 0))

print("Predictor profiles with 3 different RiskLevels:",
      inconsistent_summary.get(3, 0))

# %%
# Identify rows belonging to inconsistent predictor profiles

inconsistent_rows = df.merge(
    inconsistent_profiles.reset_index()[features.columns],
    on=list(features.columns),
    how="inner"
)

print("Rows affected by inconsistent labels:", len(inconsistent_rows))

# %%
# Examine the risk-level distribution among rows with inconsistent labels

inconsistent_rows["RiskLevel"].value_counts()

# %%
# Check class distribution after removing exact duplicate rows
df_unique = df.drop_duplicates()

print("Rows after removing exact duplicates:", len(df_unique))
print("\nRiskLevel distribution:")
print(df_unique["RiskLevel"].value_counts())

# %%
# Examine the risk-level distribution of repeated records
repeated_rows = df[df.duplicated(keep=False)]

print("Repeated rows:", len(repeated_rows))
print("\nRiskLevel distribution among repeated rows:")
print(repeated_rows["RiskLevel"].value_counts())

# %%
# Check inconsistent labels after removing exact duplicates

inconsistent_unique = (
    df_unique.groupby(list(features.columns))["RiskLevel"]
    .nunique()
)

inconsistent_unique = inconsistent_unique[
    inconsistent_unique > 1
]

print("Unique predictor profiles with inconsistent labels:",
      len(inconsistent_unique))

# %%

# Check inconsistent risk labels among unique records and examine their class distribution

inconsistent_unique_rows = df_unique.merge(
    inconsistent_unique.reset_index()[features.columns],
    on=list(features.columns),
    how="inner"
)

print("Unique rows affected by inconsistent labels:",
      len(inconsistent_unique_rows))

print("\nRiskLevel distribution:")
print(inconsistent_unique_rows["RiskLevel"].value_counts())

# %%
# Count how many inconsistent predictor profiles have each combination of risk labels
print("Number of labels per inconsistent predictor profile:")
print(inconsistent_unique.value_counts())

# %%
# Group records by predictor values and identify their distinct risk labels

inconsistent_pairs = (
    inconsistent_unique_rows
    .groupby(list(features.columns))["RiskLevel"]
    .agg(lambda x: sorted(x.unique()))
)

print(inconsistent_pairs.value_counts())

# %%
# Summarise numerical statistics for predictor values in inconsistent records

print(inconsistent_unique_rows[features.columns].describe().round(2))

# %%

# Count inconsistent records with Age below 18 and HeartRate equal to 7

print("Inconsistent rows with Age < 18:",
      (inconsistent_unique_rows["Age"] < 18).sum())

print("Inconsistent rows with HeartRate = 7:",
      (inconsistent_unique_rows["HeartRate"] == 7).sum())

# %%
# Filter inconsistent records to those aged 18 or older

inconsistent_adult_rows = inconsistent_unique_rows[
    inconsistent_unique_rows["Age"] >= 18
]

print("Inconsistent rows with Age >= 18:",
      len(inconsistent_adult_rows))

print("\nRiskLevel distribution:")
print(inconsistent_adult_rows["RiskLevel"].value_counts())

# %%
# Count predictor profiles with consistent risk labels
inconsistent_profiles = inconsistent_unique.index
consistent_profile_count = 452 - len(inconsistent_profiles)

print("Consistent predictor profiles:", consistent_profile_count)

# %%
# Count predictor profiles that have a single, consistent risk label


consistent_rows = df_unique.merge(
    inconsistent_unique.reset_index()[features.columns],
    on=list(features.columns),
    how="left",
    indicator=True
)

consistent_rows = consistent_rows[
    consistent_rows["_merge"] == "left_only"
].drop(columns=["_merge"])

print("Consistent modelling rows:", len(consistent_rows))

print("\nRiskLevel distribution:")
print(consistent_rows["RiskLevel"].value_counts())

print("\nRiskLevel percentages:")
print(
    (consistent_rows["RiskLevel"].value_counts(normalize=True) * 100)
    .round(1)
)

# %%
# Create a copy of the consistent records and check the final dataset size and risk-level distribution

df_model = consistent_rows.copy()

print("Final modelling dataset shape:", df_model.shape)
print("\nRiskLevel distribution:")
print(df_model["RiskLevel"].value_counts())

# %% [markdown]
# ## Data Quality and Cleaning Summary
#
# The original dataset contained 1,014 records and 7 variables. Initial data-quality checks found no missing values, but a substantial number of exact duplicate records were present. Removing exact duplicates reduced the dataset to 452 unique records.
#
# Further investigation identified 35 predictor profiles where the same combination of Age, SystolicBP, DiastolicBP, BS, BodyTemp and HeartRate was associated with more than one RiskLevel. Of these profiles, 34 had two different risk labels and one had all three risk labels. In total, 71 unique records were associated with these inconsistent predictor profiles.
#
# Because the correct RiskLevel could not be determined reliably from the available data, these ambiguous profiles were excluded rather than assigning or modifying their labels. This avoids introducing an unsupported assumption into the target variable.
#
# The resulting modelling dataset, `df_model`, contains 381 records with consistent predictor–target relationships.
#
# The final RiskLevel distribution is:
#
# * Low risk: 204 (53.5%)
# * Mid risk: 76 (19.9%)
# * High risk: 101 (26.5%)
#
# The original dataset has been retained unchanged, while `df_model` is used as the cleaned dataset for subsequent modelling and preprocessing.
#

# %%
# Count records with Age < 18 and examine their RiskLevel distribution

print("Records with Age < 18:", (df_model["Age"] < 18).sum())

print("\nRiskLevel distribution:")
print(df_model.loc[df_model["Age"] < 18, "RiskLevel"].value_counts())

# %%
# Display the frequency of each age below 18
print(
    df_model.loc[df_model["Age"] < 18, "Age"]
    .value_counts()
    .sort_index()
)

# %%
# Summarise the age distribution and display the frequency of each age

print(df_model["Age"].describe())

print("\nAge frequencies:")
print(df_model["Age"].value_counts().sort_index())

# %%
# Examine records aged 14 or younger and their maternal health risk levels

young_records = df_model[df_model["Age"] <= 14]

print("Records with Age <= 14:", len(young_records))

print("\nRiskLevel distribution:")
print(young_records["RiskLevel"].value_counts())

print("\nRecords:")
print(young_records[["Age", "SystolicBP", "DiastolicBP", "BS",
                     "BodyTemp", "HeartRate", "RiskLevel"]].sort_values("Age"))

# %%
# Identify and inspect records with HeartRate = 7

heart_rate_7 = df_model[df_model["HeartRate"] == 7]

print("Records with HeartRate = 7:", len(heart_rate_7))

print("\nRecords:")
print(heart_rate_7)

# %%
# Check the frequency of the specific HeartRate = 7 record in the original dataset

print("Occurrences of this exact record in original df:")
print(
    df[
        (df["Age"] == 16) &
        (df["SystolicBP"] == 120) &
        (df["DiastolicBP"] == 75) &
        (df["BS"] == 7.9) &
        (df["BodyTemp"] == 98.0) &
        (df["HeartRate"] == 7) &
        (df["RiskLevel"] == "low risk")
    ].shape[0]
)

# %%
# Exclude records with HeartRate = 7 and verify the resulting dataset
df_final = df_model[df_model["HeartRate"] != 7].copy()

print("Final modelling dataset shape:", df_final.shape)

print("\nHeartRate = 7 records:")
print((df_final["HeartRate"] == 7).sum())

print("\nRiskLevel distribution:")
print(df_final["RiskLevel"].value_counts())

# %% [markdown]
# ## Final Data Cleaning Summary
#
# The original dataset contained 1,014 records. Data-quality investigation identified substantial exact duplication and inconsistent target labels.
#
# Removing exact duplicate records resulted in 452 unique records. Further investigation identified 35 predictor profiles in which identical predictor values were associated with different `RiskLevel` labels. Because the correct label could not be determined reliably, the 71 records associated with these ambiguous profiles were excluded rather than assigning labels arbitrarily.
#
# One remaining record had an anomalous `HeartRate` value of 7 beats per minute. This record was excluded from the modelling dataset. The original dataset was retained unchanged throughout the cleaning process.
#
# Records with `Age` below 18 were retained because they were present in the source dataset with valid `RiskLevel` labels and represented all three risk categories. Their presence is treated as a characteristic and limitation of the dataset rather than a reason for automatic exclusion.
#
# The resulting `df_final` dataset contains 380 records and 7 variables, with no missing values.
#
# The final `RiskLevel` distribution is:
#
# - Low risk: 203 (53.4%)
# - Mid risk: 76 (20.0%)
# - High risk: 101 (26.6%)

# %% [markdown]
# ## 5. Prepare Features and Target
#
# The cleaned dataset, `df_final`, contains six input features and one target variable, `RiskLevel`.
# The six input features will be stored in `X`, and the target variable, `RiskLevel`, will be stored in `y`. These will be used in the subsequent model training and evaluation steps.

# %%
# Separate features and target

X = df_final.drop("RiskLevel", axis=1)
y = df_final["RiskLevel"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print(X.head())
print(y.head())

# %%
# Check the distribution of the target variable

print(y.value_counts())


# %% [markdown]
# ## 6. Encode the Target Variable
#
# The target variable, `RiskLevel`, has three ordered categories: low risk, mid risk, and high risk. These categories are encoded numerically as follows:
#
# - Low risk → 0
# - Mid risk → 1
# - High risk → 2
#
# This encoding allows the target labels to be used in model training and evaluation.

# %%
# Encode the ordinal target variable

risk_mapping = {
    "low risk": 0,
    "mid risk": 1,
    "high risk": 2
}

y_encoded = y.map(risk_mapping)

print(y_encoded.value_counts())

# %%
y_encoded


# %% [markdown]
# ## 7. Train-Test Split
#
# The dataset is divided into training and testing sets using an 80:20 split. The training set is used to fit the models, while the testing set is used to evaluate their performance on unseen data.
#
# Stratification is applied using the target variable to maintain approximately the same proportions of the three risk categories in both sets.

# %%

# %%
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# %% [markdown]
# 8. Feature Preprocessing
# The six input features have different numerical ranges and units. Their distributions and scales are examined to help determine whether feature scaling is appropriate before model training.

# %%
# Examine the range of each feature in the training data

print(X_train.describe().T)

# %% [markdown]
# ### 8.1 Feature Scaling
#
# Standardisation is applied to the numerical features using `StandardScaler`, which scales each feature based on its mean and standard deviation.
#
# The scaler is fitted using the training data only to prevent data leakage. The same fitted scaler is then used to transform both the training and testing data.

# %%

# %%
# Create the scaler

scaler = StandardScaler()

# %%
# Fit the scaler using training data and transform both datasets

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
print("Training data mean:")
print(X_train_scaled.mean(axis=0))

print("\nTraining data standard deviation:")
print(X_train_scaled.std(axis=0))

# %%

# %%
# Create the baseline Logistic Regression model

baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# %%
# Train the model using the scaled training data

baseline_model.fit(X_train_scaled, y_train)

# %%
# Make predictions on the test data

y_pred = baseline_model.predict(X_test_scaled)

print(y_pred)

# %%
# Evaluate the baseline model on the test data


print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["low risk", "mid risk", "high risk"]
))

print("Balanced Accuracy:")
print(balanced_accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# ### 8.2 Baseline Model: Logistic Regression
#
# A multinomial Logistic Regression model was trained as the baseline classifier using the standardised training features.
#
# The model achieved 66% accuracy and 54.3% balanced accuracy on the test set. Performance was strongest for the low-risk class, with 88% recall, while the mid-risk class had substantially lower recall (20%) and an F1-score of 0.26.
#
# The baseline provides a reference point for evaluating the performance of other classification models.

# %%

# %%
# Create the Random Forest model

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# %%
# Train the Random Forest model

rf_model.fit(X_train, y_train)

# %%
# Generate Random Forest predictions
y_pred_rf = rf_model.predict(X_test)

# %%

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred_rf,
    target_names=["low risk", "mid risk", "high risk"]
))

print("Balanced Accuracy:")
print(balanced_accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# %% [markdown]
# ### 8.3 Random Forest Model
#
# A Random Forest classifier with 100 decision trees was trained using the unscaled training features.
#
# The model achieved 75% accuracy and 65.6% balanced accuracy on the test set, outperforming the Logistic Regression baseline on both metrics. Recall was 90% for low-risk cases and 80% for high-risk cases. However, mid-risk recall remained low at 27%, with only 4 of the 15 actual mid-risk cases correctly classified.
#
# These results indicate improved overall classification performance compared with the baseline, although identifying mid-risk cases remains a challenge.

# %%

# %%
# Create the XGBoost model

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)

# %%
# Train the XGBoost model

xgb_model.fit(X_train, y_train)

# %%
# Generate predictions using the trained XGBoost model
y_pred_xgb = xgb_model.predict(X_test)

# %% [markdown]
# ### Evaluation of the Final XGBoost Model
#
# The final XGBoost model was evaluated on the held-out test set using a classification report, balanced accuracy and a confusion matrix. The classification report provides precision, recall and F1-scores for each risk category, while balanced accuracy accounts for differences in class sizes. The confusion matrix shows the number of correct and incorrect predictions for each risk category, helping identify which classes are more difficult for the model to classify.

# %%

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred_xgb,
    target_names=["low risk", "mid risk", "high risk"]
))

print("Balanced Accuracy:")
print(balanced_accuracy_score(y_test, y_pred_xgb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# %% [markdown]
# ### 8.4 XGBoost Model
#
# An XGBoost classifier was trained using 100 boosting rounds, a maximum tree depth of 3, and a learning rate of 0.1.
#
# The model achieved 82.9% accuracy and 75.6% balanced accuracy on the test set, outperforming the Logistic Regression and Random Forest models on both metrics. Recall was 95% for low-risk cases, 47% for mid-risk cases, and 85% for high-risk cases.
#
# Although XGBoost showed the strongest overall performance among the three models on the test set, mid-risk cases remained more difficult to classify. These findings should be interpreted cautiously because the evaluation used a test set of 76 records.

# %%

# %%
# Set up 5-fold stratified cross-validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# %%
# Evaluate XGBoost using 5-fold cross-validation

cv_scores_xgb = cross_val_score(
    xgb_model,
    X_train,
    y_train,
    cv=cv,
    scoring="balanced_accuracy"
)

print("Cross-validation scores:")
print(cv_scores_xgb)

print("\nMean balanced accuracy:")
print(cv_scores_xgb.mean())

print("\nStandard deviation:")
print(cv_scores_xgb.std())

# %% [markdown]
# ### 8.5 XGBoost Cross-Validation
#
# XGBoost was evaluated using five-fold stratified cross-validation on the training data, with balanced accuracy as the scoring metric.
#
# The mean balanced accuracy across the five folds was 66.0%, with a standard deviation of 1.9 percentage points. The fold scores ranged from 64.1% to 69.4%, indicating relatively consistent performance across the validation folds.
#
# The cross-validation mean was lower than the 75.6% balanced accuracy achieved on the held-out test set. This difference suggests that the test-set result may not fully reflect the model's typical performance, so the findings should be interpreted cautiously.

# %%
# Evaluate Random Forest using 5-fold cross-validation

cv_scores_rf = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=cv,
    scoring="balanced_accuracy"
)

print("Cross-validation scores:")
print(cv_scores_rf)

print("\nMean balanced accuracy:")
print(cv_scores_rf.mean())

print("\nStandard deviation:")
print(cv_scores_rf.std())

# %%

# Evaluate Logistic Regression using 5-fold cross-validation
# Scaling is fitted separately within each training fold


baseline_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        random_state=42
    )
)

cv_scores_lr = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=cv,
    scoring="balanced_accuracy"
)

print("Cross-validation scores:")
print(cv_scores_lr)

print("\nMean balanced accuracy:")
print(cv_scores_lr.mean())

print("\nStandard deviation:")
print(cv_scores_lr.std())


# %% [markdown]
# ### **Logistic Regression Cross-Validation**
#
# Logistic Regression was evaluated using stratified 5-fold cross-validation, with balanced accuracy as the evaluation metric. A pipeline was used to fit feature scaling separately within each fold, reducing the risk of data leakage.
#
# The model achieved a mean balanced accuracy of **59.08%**, with a standard deviation of **9.47 percentage points** across the five folds. The variation in fold scores indicates that performance differed across the validation subsets.

# %% [markdown]
# ### 8.6 Model Validation
#
# Five-fold stratified cross-validation was performed on the training data using balanced accuracy to assess model performance across different training and validation subsets. For Logistic Regression, feature scaling was fitted separately within each fold using a pipeline.
#
# The mean cross-validation balanced accuracy was 59.08% for Logistic Regression, 67.14% for Random Forest, and 65.99% for XGBoost. Random Forest achieved the highest mean cross-validation score. Logistic Regression showed the greatest variation across folds, while XGBoost showed the lowest variation.
#
# On the separate held-out test set, XGBoost achieved the highest balanced accuracy at 75.6%. The difference between cross-validation and held-out test-set results illustrates that model performance can vary accross evaluation subsets.
#
# These results suggest that Random Forest and XGBoost performed better than the Logistic Regression baseline, although their relative performance differed between cross-validation and held-out testing. Further model analysis and explainability were used to inform the selection of a model for the prototype.

# %%

print("SHAP version:", shap.__version__)

# %%
# Create a SHAP TreeExplainer for the trained XGBoost model


explainer_xgb = shap.TreeExplainer(xgb_model)

print("SHAP explainer created successfully.")

# %%
# Calculate SHAP values for the test data

shap_values_xgb = explainer_xgb.shap_values(X_test)

print("SHAP values calculated successfully.")
print("SHAP values shape:", shap_values_xgb.shape)

# %%
# Calculate mean absolute SHAP values across test records and risk classes
mean_abs_shap = np.abs(shap_values_xgb).mean(axis=(0, 2))

shap_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean |SHAP value|": mean_abs_shap
}).sort_values("Mean |SHAP value|", ascending=False)

print("Overall SHAP feature importance:")
print(shap_importance)

# %% [markdown]
# The mean absolute SHAP values were calculated across the test records and all three risk classes to estimate the overall contribution of each feature to the model's predictions.
#
# Blood sugar (`BS`) had the highest mean absolute SHAP value (0.8853), followed by `SystolicBP` (0.4959) and `BodyTemp` (0.3860). `Age`, `DiastolicBP`, and `HeartRate` had lower values.
#
# These results indicate which features contributed most strongly to the model's predictions overall. However, mean absolute SHAP values do not indicate the direction of a feature's effect or establish a causal relationship with maternal health risk.

# %%
# SHAP summary plot for High Risk (class 2)

shap.summary_plot(
    shap_values_xgb[:, :, 2],
    X_test,
    feature_names=X_test.columns
)

# %%
# SHAP summary plot for low risk (class 0)
shap.summary_plot(
    shap_values_xgb[:, :, 0],
    X_test,
    feature_names=X_test.columns
)

# %%
# SHAP summary plot for mid risk (class 1)

shap.summary_plot(
    shap_values_xgb[:, :, 1],
    X_test,
    feature_names=X_test.columns
)

# %% [markdown]
# ## SHAP Explainability
#
# SHAP (SHapley Additive exPlanations) was used to understand how individual features contribute to the XGBoost model's predictions for each maternal health risk class.
#
# Across the three risk classes, **BS, SystolicBP and BodyTemp** showed relatively strong influence on model predictions. For Low and High Risk, these features showed clearer directional patterns, with lower values generally contributing toward Low Risk and higher values generally contributing toward High Risk.
#
# The Mid Risk class showed more mixed feature effects, suggesting that predictions for this class may depend more on combinations of features rather than a single feature.
#
# **Age, DiastolicBP and HeartRate** generally showed smaller contributions compared with the leading features.
#
# The SHAP analysis provides additional interpretability of the XGBoost model and will be considered together with model performance and cross-validation results when selecting the model for the prototype.

# %%

# Recreate Logistic Regression pipeline with feature scaling for later comparison

# Train Logistic Regression with feature scaling
logreg_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42
    )
)

logreg_model.fit(X_train, y_train)

# Generate predictions on the held-out test set
y_pred_logreg = logreg_model.predict(X_test)

print("Scaled Logistic Regression model trained successfully.")

# %%
# Recreate the Random Forest model for later comparison


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest model recreated successfully.")

# %%

# Generate XGBoost predictions before evaluating the models
y_pred_xgb = xgb_model.predict(X_test)

# Calculate performance for each model
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

logreg_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_logreg)
rf_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_rf)
xgb_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_xgb)

print("Logistic Regression")
print("Accuracy:", logreg_accuracy)
print("Balanced Accuracy:", logreg_balanced_accuracy)

print("\nRandom Forest")
print("Accuracy:", rf_accuracy)
print("Balanced Accuracy:", rf_balanced_accuracy)

print("\nXGBoost")
print("Accuracy:", xgb_accuracy)
print("Balanced Accuracy:", xgb_balanced_accuracy)

# %%
# Evaluate Random Forest using 5-fold cross-validation

cv_scores_rf = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=cv,
    scoring="balanced_accuracy"
)

print("Cross-validation scores:")
print(cv_scores_rf)

print("\nMean balanced accuracy:")
print(cv_scores_rf.mean())

print("\nStandard deviation:")
print(cv_scores_rf.std())

# %%
# Create a scaled Logistic Regression pipeline for cross-validation

lr_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42
    )
)

cv_scores_lr = cross_val_score(
    lr_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="balanced_accuracy"
)

print("Cross-validation scores:")
print(cv_scores_lr)

print("\nMean balanced accuracy:")
print(cv_scores_lr.mean())

print("\nStandard deviation:")
print(cv_scores_lr.std())

# %% [markdown]
# Logistic Regression achieved a mean cross-validation balanced accuracy of **59.08%**, with a standard deviation of **9.47 percentage points** across the five folds. The relatively high variation indicates that its performance differed across the folds. The mean cross-validation score was higher than the held-out test balanced accuracy of **54.3%**, illustrating variation between cross-validation and test-set performance.

# %%
scaler = StandardScaler()

# %%
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
print("Logistic Regression Classification Report")
print(classification_report(
    y_test,
    y_pred_logreg,
    target_names=["Low Risk", "Mid Risk", "High Risk"]
))

# %% [markdown]
# ### **Model Confusion Matrices**
#
# Confusion matrices compare the actual risk categories with the categories predicted by each model. They show the number of correct classifications and the cases misclassified into other risk categories. The matrices below are used to examine the classification errors made by Logistic Regression, Random Forest, and XGBoost on the held-out test set.

# %%

cm_xgb = confusion_matrix(y_test, y_pred_xgb)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_xgb,
    display_labels=["Low Risk", "Mid Risk", "High Risk"]
)

disp.plot()
plt.title("XGBoost Confusion Matrix")
plt.show()

# %%
cm_rf = confusion_matrix(y_test, y_pred_rf)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Low Risk", "Mid Risk", "High Risk"]
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

# %%
cm_lr = confusion_matrix(y_test, y_pred_logreg)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Low Risk", "Mid Risk", "High Risk"]
)

disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# %%
# Model Performance Comparison

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Test Accuracy": [
        logreg_accuracy,
        rf_accuracy,
        xgb_accuracy
    ],
    "Test Balanced Accuracy": [
        logreg_balanced_accuracy,
        rf_balanced_accuracy,
        xgb_balanced_accuracy
    ],
    "CV Mean Balanced Accuracy": [
        cv_scores_lr.mean(),
        cv_scores_rf.mean(),
        cv_scores_xgb.mean()
    ],
    "CV Standard Deviation": [
        cv_scores_lr.std(),
        cv_scores_rf.std(),
        cv_scores_xgb.std()
    ]
})

print(model_comparison.round(3))

# %% [markdown]
# ### **Model Performance Comparison**
#
# The three models were evaluated using test-set accuracy, test-set balanced accuracy, and 5-fold cross-validation using balanced accuracy. Balanced accuracy was used because the three risk classes are not equally represented in the dataset.
#
# Logistic Regression provided the baseline performance, with a mean cross-validation balanced accuracy of **59.1%**. Random Forest achieved the highest mean cross-validation balanced accuracy at 67.1%, followed by XGBoost at 66.0%.
#
# On the held-out test set, XGBoost achieved the highest accuracy (82.9%) and balanced accuracy (75.6%). It also achieved higher recall for the Mid Risk class (**47%**) than Random Forest (27%) and Logistic Regression (20%).
#
# The standard deviations of the cross-validation balanced accuracy scores were **9.5 percentage points for Logistic Regression, 4.2 percentage points for Random Forest, and 1.9 percentage points for XGBoost**. Model selection therefore considers overall test performance, class-level performance, cross-validation results, and SHAP-based interpretability rather than relying on a single metric.

# %%
# Class-level Recall Comparison

class_recall_comparison = pd.DataFrame({
    "Risk Class": ["Low Risk", "Mid Risk", "High Risk"],
    "Logistic Regression": [
        0.88,
        0.20,
        0.55
    ],
    "Random Forest": [
        0.90,
        0.27,
        0.80
    ],
    "XGBoost": [
        0.95,
        0.47,
        0.85
    ]
})

print(class_recall_comparison.round(2))
# %% [markdown]
# ### **Class-Level Recall Comparison**
#
# Recall was also examined for each risk class because overall accuracy can hide differences in how well the model identifies individual classes.
#
# XGBoost achieved the highest recall for all three classes on the held-out test set, with recall of 95% for Low Risk, 47% for Mid Risk, and 85% for High Risk. Random Forest achieved 90%, 27%, and 80% respectively, while Logistic Regression achieved **88%**, 20%, and 55%.
#
# The Mid Risk class was the most difficult class for all three models. Although XGBoost performed better than the other models for this class, its recall of 47% indicates that a substantial proportion of Mid Risk cases were still misclassified.
#
# These class-level results complement the cross-validation and SHAP analyses and highlight the importance of considering performance for each risk class when assessing the suitability of the prototype model.

# %% [markdown]
# ## **Final Model Selection**
#
# Based on the model evaluation, **XGBoost was selected as the final model for the prototype**.
#
# XGBoost achieved the highest performance on the held-out test set, with an accuracy of 82.9% and balanced accuracy of 75.6%. It also achieved the highest recall across all three risk classes: 95% for Low Risk, 47% for Mid Risk, and 85% for High Risk.
#
# The cross-validation results differed from the held-out test results. Random Forest achieved a slightly higher mean cross-validation balanced accuracy of 67.1%, compared with 66.0% for XGBoost. Random Forest also showed lower variation across folds, with a standard deviation of 4.2 percentage points compared with 1.9 percentage points for XGBoost. These differences were considered when selecting the final model rather than relying solely on test-set performance.
#
# XGBoost was selected because of its stronger held-out test performance and higher recall for each risk class compared with the other models. However, its 47% recall for Mid Risk indicates that more than half of the actual Mid Risk cases were misclassified. SHAP analysis provided additional insight into the model's predictions, identifying BS, SystolicBP, and BodyTemp as the most influential features overall.
#
# The selected XGBoost model is intended for **prototype and educational purposes**, not clinical diagnosis. The findings should be interpreted cautiously because the final dataset is relatively small and the model has not been externally validated. Larger, more representative datasets and further validation would be needed before considering real-world clinical use.

# %%

# Save the final XGBoost model
joblib.dump(xgb_model, "xgb_final_model.pkl")

print("Final XGBoost model saved successfully.")

# %% [markdown]
# ### Verify the Saved Model
#
# The saved XGBoost model was reloaded and used to generate predictions on the test set. These predictions were compared with those from the original model to verify that the saved model retained the same prediction behaviour.

# %%
# Load the saved final model
loaded_model = joblib.load("xgb_final_model.pkl")

print("Saved XGBoost model loaded successfully.")

# %%
# Test the loaded model on the test data

loaded_predictions = loaded_model.predict(X_test)

print("Predictions generated successfully.")
print("Number of predictions:", len(loaded_predictions))

# %%
# Check that the saved model produces the same predictions
# as the original trained XGBoost model

predictions_match = np.array_equal(
    y_pred_xgb,
    loaded_predictions
)

print("Predictions match:", predictions_match)

# %% [markdown]
# ### Prototype Prediction Function
#
# A reusable function was created to accept six maternal health input features and return the predicted risk category using the saved XGBoost model. The numerical prediction is mapped back to a readable label: Low Risk, Mid Risk, or High Risk.

# %%
# Prototype prediction function

risk_labels = {
    0: "Low Risk",
    1: "Mid Risk",
    2: "High Risk"
}


def predict_maternal_risk(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    input_data = pd.DataFrame([{
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": bs,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate
    }])

    prediction = loaded_model.predict(input_data)[0]

    return risk_labels[prediction]

# %% [markdown]
# ### Test the Prototype
#
# The prediction function was tested using an example set of maternal health feature values to verify that it accepts the inputs and returns a readable risk category.

# %%
# Test the prototype prediction function


result = predict_maternal_risk(
    age=25,
    systolic_bp=120,
    diastolic_bp=80,
    bs=6.0,
    body_temp=98.0,
    heart_rate=75
)

print("Predicted Risk Level:", result)

# %% [markdown]
# ### Test Predictions on Held-out Examples
#
# The saved XGBoost model was used to predict risk categories for records from the held-out test set. The predicted categories were compared with the actual risk labels to illustrate the prototype's predictions and identify examples where the model's predictions differed from the actual labels.

# %%
# Test the prototype using examples from the test set

test_examples = X_test.copy()
test_examples["Actual Risk"] = y_test.map(risk_labels)

test_examples["Predicted Risk"] = loaded_model.predict(X_test)
test_examples["Predicted Risk"] = test_examples["Predicted Risk"].map(
    risk_labels)

print(test_examples.head(10))
# %%
# Show prediction probabilities for the first test example

sample = X_test.iloc[[0]]

probabilities = loaded_model.predict_proba(sample)[0]

probability_table = pd.DataFrame({
    "Risk Level": ["Low Risk", "Mid Risk", "High Risk"],
    "Probability": probabilities
})

probability_table["Probability"] = (
    probability_table["Probability"] * 100
).round(2)

print(probability_table)
# %% [markdown]
# ### Prototype Prediction with Probabilities
#
# The prediction function was extended to return the predicted risk category alongside the model's estimated probabilities for all three categories. These probabilities provide additional information about the model's output but should not be interpreted as verified clinical risk probabilities.

# %%
# Prototype prediction function with probabilities


def predict_maternal_risk(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    input_data = pd.DataFrame([{
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": bs,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate
    }])

    prediction = loaded_model.predict(input_data)[0]
    probabilities = loaded_model.predict_proba(input_data)[0]

    probability_table = pd.DataFrame({
        "Risk Level": ["Low Risk", "Mid Risk", "High Risk"],
        "Model Probability (%)": (probabilities * 100).round(2)
    })

    return risk_labels[prediction], probability_table

# %% [markdown]
# ### Test the Improved Prototype
#
# The updated prediction function was tested using example input values. The output includes the predicted risk category and the model's estimated probabilities for Low Risk, Mid Risk, and High Risk.

# %%
# Test the improved prototype


predicted_risk, probabilities = predict_maternal_risk(
    age=25,
    systolic_bp=120,
    diastolic_bp=80,
    bs=6.0,
    body_temp=98.0,
    heart_rate=75
)

print("Predicted Risk Level:", predicted_risk)
print(probabilities)
# %%
print(loaded_model.classes_)

# %% [markdown]
# **Verification of class order:** The model's output classes were checked to confirm that the probability columns correspond to the intended risk categories: 0 = Low Risk, 1 = Mid Risk, and 2 = High Risk.

# %% [markdown]
# ### Prototype Prediction Output
#
# The prototype returns a predicted maternal health risk category along with model-estimated probabilities for each risk class.
#
# These probabilities represent the outputs of the XGBoost model and should not be interpreted as clinically validated probabilities of maternal health risk. The prototype is intended for educational and demonstration purposes only; it is not a diagnostic tool or a substitute for clinical judgement.

# %% [markdown]
# ### Example Input
#
# An example set of maternal health feature values was prepared in a DataFrame to demonstrate how the prototype accepts input data.

# %%
# Example patient input

patient = {
    "Age": 25,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 6.0,
    "BodyTemp": 98.0,
    "HeartRate": 75
}

patient_df = pd.DataFrame([patient])

print(patient_df)
# %% [markdown]
# # Test the prototype with a second example
#
# example_patient_2 = {
#     "Age": 42,
#     "SystolicBP": 140,
#     "DiastolicBP": 100,
#     "BS": 18.0,
#     "BodyTemp": 98.0,
#     "HeartRate": 90
# }
#
# example_patient_2_df = pd.DataFrame([example_patient_2])
#
# # Generate prediction and probabilities
# example_prediction = loaded_model.predict(example_patient_2_df)[0]
# example_probabilities = loaded_model.predict_proba(example_patient_2_df)[0]
#
# print("Maternal Health Risk Prediction — Example 2")
# print("-------------------------------------------")
# print(f"Predicted Risk Level: {risk_labels[example_prediction]}")
# print()
#
# # Display model-estimated probabilities
# for label, probability in zip(
#     ["Low Risk", "Mid Risk", "High Risk"],
#     example_probabilities
# ):
#     print(f"{label}: {probability * 100:.2f}%")

# %% [markdown]
# ### Prototype Demonstration with Different Input Scenarios
#
# Two example input scenarios were used to demonstrate how the prototype generates risk-category predictions and model-estimated probabilities for different combinations of maternal health features. These examples illustrate the prototype's functionality and do not establish clinical validity.

# %%
# Test the prototype with a second example

example_patient_2 = {
    "Age": 42,
    "SystolicBP": 140,
    "DiastolicBP": 100,
    "BS": 18.0,
    "BodyTemp": 98.0,
    "HeartRate": 90
}

example_patient_2_df = pd.DataFrame([example_patient_2])

# Generate prediction and probabilities
example_prediction = loaded_model.predict(example_patient_2_df)[0]
example_probabilities = loaded_model.predict_proba(example_patient_2_df)[0]

print("Maternal Health Risk Prediction — Example 2")
print("-------------------------------------------")
print(f"Predicted Risk Level: {risk_labels[example_prediction]}")
print()

# Display model-estimated probabilities
for label, probability in zip(
    ["Low Risk", "Mid Risk", "High Risk"],
    example_probabilities
):
    print(f"{label}: {probability * 100:.2f}%")

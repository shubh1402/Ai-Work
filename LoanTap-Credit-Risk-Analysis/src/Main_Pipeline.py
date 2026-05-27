# =========================================
# LOANTAP CREDIT RISK ANALYSIS
# =========================================

# =========================================
# IMPORT REQUIRED LIBRARIES
# =========================================

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("../datasets/LoanTapData.csv")

# =========================================
# BASIC DATA EXPLORATION
# =========================================

print("\n================ DATASET INFO ================\n")

print(df.info())




print("\n================ FIRST 5 ROWS ================\n")

print(df.head())


print("\n================ STATISTICAL SUMMARY ================\n")

print(df.describe())


# =========================================
# CHECK MISSING VALUES
# =========================================

print("\n================ MISSING VALUES ================\n")

print(df.isnull().sum())


# =========================================
# TARGET VARIABLE DISTRIBUTION
# =========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    x='loan_status',
    data=df
)

plt.title("Loan Status Distribution")

plt.show()


# =========================================
# FEATURE ENGINEERING
# =========================================

# Convert Loan Status into Binary Classification

df['loan_status'] = df['loan_status'].apply(
    lambda x: 1 if x == 'Fully Paid' else 0
)


# Public Record Flag

df['pub_rec_flag'] = df['pub_rec'].apply(
    lambda x: 1 if x > 0 else 0
)


# Mortgage Account Flag

df['mort_acc_flag'] = df['mort_acc'].apply(
    lambda x: 1 if x > 0 else 0
)


# Bankruptcy Flag

df['pub_rec_bankruptcies_flag'] = df[
    'pub_rec_bankruptcies'
].apply(
    lambda x: 1 if x > 0 else 0
)


print("\n================ FEATURE ENGINEERING COMPLETE ================\n")

print(df[
    [
        'pub_rec',
        'pub_rec_flag',
        'mort_acc',
        'mort_acc_flag',
        'pub_rec_bankruptcies',
        'pub_rec_bankruptcies_flag'
    ]
].head())


# =========================================
# MISSING VALUE TREATMENT
# =========================================

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100


print("\n================ MISSING VALUE PERCENTAGE ================\n")

print(missing_percentage.sort_values(ascending=False))


# Drop columns with excessive missing values

df = df.drop(
    columns=[
        'emp_title',
        'title',
        'Address'
    ],
    errors='ignore'
)


# Fill Numerical Missing Values

numerical_columns = df.select_dtypes(
    include=np.number
).columns


for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())


# =========================================
# OUTLIER ANALYSIS
# =========================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df['loan_amnt']
)

plt.title("Loan Amount Outlier Analysis")

plt.show()


plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df['annual_inc']
)

plt.title("Annual Income Outlier Analysis")

plt.show()


# =========================================
# FEATURE SELECTION
# =========================================

selected_features = [
    'loan_amnt',
    'int_rate',
    'installment',
    'annual_inc',
    'dti',
    'open_acc',
    'pub_rec',
    'revol_bal',
    'revol_util',
    'total_acc',
    'mort_acc',
    'pub_rec_bankruptcies',
    'pub_rec_flag',
    'mort_acc_flag',
    'pub_rec_bankruptcies_flag'
]


# Remove rows with missing target values

df = df.dropna(subset=['loan_status'])


X = df[selected_features]

y = df['loan_status']


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n================ TRAIN TEST SPLIT ================\n")

print("Training Shape:", X_train.shape)

print("Testing Shape:", X_test.shape)


# =========================================
# FEATURE SCALING
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================
# LOGISTIC REGRESSION MODEL
# =========================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)


# Predictions

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:, 1]


print("\n================ MODEL TRAINING COMPLETE ================\n")


# =========================================
# MODEL EVALUATION
# =========================================

print("\n================ CLASSIFICATION REPORT ================\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# Confusion Matrix

conf_matrix = confusion_matrix(
    y_test,
    y_pred
)

print("\n================ CONFUSION MATRIX ================\n")

print(conf_matrix)


# ROC AUC Score

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n================ ROC AUC SCORE ================\n")

print(roc_auc)


# =========================================
# ROC CURVE
# =========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label=f'ROC Curve (AUC = {roc_auc:.2f})'
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC AUC Curve")

plt.legend()

plt.show()


# =========================================
# PRECISION RECALL CURVE
# =========================================

precision, recall, pr_thresholds = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 5))

plt.plot(
    recall,
    precision
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.show()


# =========================================
# MODEL COEFFICIENTS
# =========================================

coefficients = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': model.coef_[0]
})

coefficients = coefficients.sort_values(
    by='Coefficient',
    ascending=False
)

print("\n================ MODEL COEFFICIENTS ================\n")

print(coefficients)


# =========================================
# BUSINESS INSIGHTS
# =========================================

print("\n================ BUSINESS INSIGHTS ================\n")

print("""
1. Customers with stronger financial indicators showed higher probability
   of fully repaying loans.

2. Public records, bankruptcies, and high debt-to-income ratios negatively
   impacted loan repayment behavior.

3. Logistic Regression successfully identified patterns associated with
   credit risk and repayment probability.

4. ROC AUC analysis demonstrated the model’s ability to distinguish
   between low-risk and high-risk borrowers.

5. Precision-Recall tradeoff is critical in loan underwriting because:
   - Higher Recall helps detect more real defaulters
   - Higher Precision reduces rejection of good borrowers

6. Financial institutions must balance loan approval growth with
   NPA (Non-Performing Asset) prevention strategies.
""")

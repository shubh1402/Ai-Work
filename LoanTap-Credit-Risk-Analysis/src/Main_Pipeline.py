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

df = pd.read_csv("LoanTapData.csv")


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

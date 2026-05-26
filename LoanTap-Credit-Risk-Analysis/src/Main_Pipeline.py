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

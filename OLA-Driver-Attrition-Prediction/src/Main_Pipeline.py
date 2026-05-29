# =========================================
# OLA DRIVER ATTRITION PREDICTION
# =========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer

import warnings
warnings.filterwarnings("ignore")


# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv("../datasets/ola_driver_scaler.csv")

print(df.shape)
print(df.head())
print(df.info())


# =========================================
# DATE CONVERSION
# =========================================

df["MMM-YY"] = pd.to_datetime(df["MMM-YY"])

df["Dateofjoining"] = pd.to_datetime(
    df["Dateofjoining"]
)

df["LastWorkingDate"] = pd.to_datetime(
    df["LastWorkingDate"]
)


# =========================================
# MISSING VALUES
# =========================================

print(df.isnull().sum())


# =========================================
# KNN IMPUTATION
# =========================================

numeric_cols = [
    "Age",
    "Income",
    "Total Business Value",
    "Quarterly Rating",
    "Grade"
]

imputer = KNNImputer(
    n_neighbors=5
)

df[numeric_cols] = imputer.fit_transform(
    df[numeric_cols]
)


# =========================================
# DRIVER LEVEL AGGREGATION
# =========================================

driver_df = pd.DataFrame()

driver_df["Driver_ID"] = df["Driver_ID"].unique()

driver_df.set_index(
    "Driver_ID",
    inplace=True
)


# =========================================
# LAST RECORD FEATURES
# =========================================

driver_df["Age"] = (
    df.groupby("Driver_ID")["Age"]
    .last()
)

driver_df["Gender"] = (
    df.groupby("Driver_ID")["Gender"]
    .last()
)

driver_df["City"] = (
    df.groupby("Driver_ID")["City"]
    .last()
)

driver_df["Education_Level"] = (
    df.groupby("Driver_ID")["Education_Level"]
    .last()
)

driver_df["Income"] = (
    df.groupby("Driver_ID")["Income"]
    .last()
)

driver_df["Grade"] = (
    df.groupby("Driver_ID")["Grade"]
    .last()
)

driver_df["Joining_Designation"] = (
    df.groupby("Driver_ID")
    ["Joining Designation"]
    .last()
)

driver_df["Quarterly_Rating"] = (
    df.groupby("Driver_ID")
    ["Quarterly Rating"]
    .last()
)

driver_df["Total_Business_Value"] = (
    df.groupby("Driver_ID")
    ["Total Business Value"]
    .sum()
)


# =========================================
# TARGET VARIABLE
# =========================================

driver_df["Target"] = (
    df.groupby("Driver_ID")
    ["LastWorkingDate"]
    .last()
    .notnull()
    .astype(int)
)


# =========================================
# RATING INCREASE FEATURE
# =========================================

rating = (
    df.groupby("Driver_ID")
    ["Quarterly Rating"]
    .agg(["first","last"])
)

driver_df["Rating_Increased"] = (
    rating["last"] >
    rating["first"]
).astype(int)


# =========================================
# INCOME INCREASE FEATURE
# =========================================

income = (
    df.groupby("Driver_ID")
    ["Income"]
    .agg(["first","last"])
)

driver_df["Income_Increased"] = (
    income["last"] >
    income["first"]
).astype(int)


print(driver_df.head())

print(driver_df.shape)
# =========================================
# CORRELATION HEATMAP
# =========================================

plt.figure(figsize=(10,8))

sns.heatmap(
    driver_df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "../screenshots/Correlation_Heatmap.png"
)

plt.close()


# =========================================
# ONE HOT ENCODING
# =========================================

driver_df = pd.get_dummies(
    driver_df,
    columns=["City"],
    drop_first=True
)


# =========================================
# TRAIN TEST SPLIT
# =========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = driver_df.drop(
    "Target",
    axis=1
)

y = driver_df["Target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================
# STANDARDIZATION
# =========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# =========================================
# RANDOM FOREST
# =========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:,1]

print("\nRANDOM FOREST")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        rf_prob
    )
)


# =========================================
# XGBOOST
# =========================================

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb.fit(
    X_train,
    y_train
)

xgb_pred = xgb.predict(X_test)

xgb_prob = xgb.predict_proba(X_test)[:,1]

print("\nXGBOOST")

print(
    classification_report(
        y_test,
        xgb_pred
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        xgb_prob
    )
)


# =========================================
# ROC CURVE
# =========================================

fpr,tpr,_ = roc_curve(
    y_test,
    xgb_prob
)

plt.figure(figsize=(8,5))

plt.plot(
    fpr,
    tpr
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.savefig(
    "../screenshots/ROC_Curve.png"
)

plt.close()


# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,8))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance.head(15)
)

plt.title("Top Feature Importance")

plt.savefig(
    "../screenshots/Feature_Importance.png"
)

plt.close()


# =========================================
# EXPORT
# =========================================

importance.to_csv(
    "../results/feature_importance.csv",
    index=False
)

driver_df.to_csv(
    "../results/final_driver_dataset.csv"
)

print("\n===================================")
print("OLA PIPELINE COMPLETED")
print("===================================")
# OLA Driver Attrition Prediction Using Ensemble Learning

## Project Overview

This project focuses on predicting driver attrition for Ola using machine learning and ensemble learning techniques.

Driver churn is one of the biggest challenges in the ride-hailing industry. Recruiting new drivers is significantly more expensive than retaining existing drivers. This analysis aims to identify the key factors influencing driver attrition and build predictive models that can help Ola proactively retain valuable drivers.

---

# Business Problem

Recruiting and retaining drivers is a major operational challenge for Ola.

High driver attrition leads to:

* Increased driver acquisition costs
* Reduced service availability
* Lower operational efficiency
* Increased business uncertainty

The objective of this project is to predict whether a driver is likely to leave the company using demographic, performance, income, and tenure-related information.

---

# Dataset Description

The dataset contains monthly driver information including:

* Driver Demographics
* Age
* Gender
* City
* Education Level
* Monthly Income
* Joining Designation
* Grade
* Quarterly Rating
* Total Business Value
* Joining Date
* Last Working Date

---

# Data Preprocessing

The following preprocessing techniques were applied:

* Date Feature Conversion
* Missing Value Analysis
* KNN Imputation
* Driver-Level Aggregation
* Feature Engineering
* One-Hot Encoding
* Feature Scaling
* Correlation Analysis

---

# Feature Engineering

Additional business-driven features were created:

### Target Variable

A driver is considered churned if a Last Working Date is present.

### Rating Improvement Flag

Indicates whether a driver's quarterly rating increased over time.

### Income Improvement Flag

Indicates whether a driver's income increased during their tenure.

### Driver-Level Aggregation

Monthly records were consolidated into a single record per driver for modeling purposes.

---

# Machine Learning Models

## Random Forest Classifier (Bagging)

Used to capture complex relationships between driver attributes and attrition behavior.

### Benefits

* Handles non-linearity
* Reduces overfitting
* Provides feature importance

---

## XGBoost Classifier (Boosting)

Used to improve prediction performance through gradient boosting.

### Benefits

* High predictive performance
* Handles feature interactions effectively
* Robust to noisy data

---

# Model Evaluation

The models were evaluated using:

* Classification Report
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* ROC Curve Analysis

---

# Visualizations Generated

## Correlation Heatmap

Analyzes relationships between important driver attributes.

## ROC Curve

Evaluates model discrimination capability.

## Feature Importance

Identifies the most influential factors affecting driver attrition.

---

# Key Business Insights

* Income growth significantly impacts driver retention.
* Improvement in quarterly ratings is associated with lower attrition.
* Total Business Value is a strong indicator of driver engagement.
* Drivers with stagnant income and performance are more likely to leave.
* Driver retention strategies should focus on performance incentives and income growth opportunities.

---

# Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Imbalanced-Learn

---

# Project Structure

```bash
OLA-Driver-Attrition-Prediction/
│
├── datasets/
│   └── ola_driver_scaler.csv
│
├── screenshots/
│   ├── Correlation_Heatmap.png
│   ├── ROC_Curve.png
│   └── Feature_Importance.png
│
├── results/
│   ├── feature_importance.csv
│   └── final_driver_dataset.csv
│
├── src/
│   └── Main_Pipeline.py
│
├── visualizations/
├── notebooks/
├── README.md
└── requirements.txt
```

---

# Skills Demonstrated

* Data Cleaning
* Feature Engineering
* KNN Imputation
* Ensemble Learning
* Random Forest
* XGBoost
* Classification Modeling
* Driver Attrition Analytics
* Business Intelligence
* Predictive Analytics

---

# Project Status

✅ Completed Production-Style Machine Learning Project

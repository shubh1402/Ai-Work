# =========================================
# WALMART STATISTICAL ANALYSIS
# =========================================

# Import Required Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
import statsmodels.api as sm


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("../datasets/Walmart_data.csv")


# =========================================
# BASIC DATA EXPLORATION
# =========================================

print(df.info())

print(df.head())

print(df.describe())


# =========================================
# CONVERT CATEGORICAL COLUMNS
# =========================================

categorical_columns = [
    'Gender',
    'Age',
    'City_Category',
    'Marital_Status'
]

for col in categorical_columns:
    df[col] = df[col].astype('category')


# =========================================
# CHECK MISSING VALUES
# =========================================

print(df.isnull().sum())


# =========================================
# OUTLIER ANALYSIS
# =========================================

sns.boxplot(x='Gender', y='Purchase', data=df)

plt.title("Outliers in Purchase Amount by Gender")

plt.show()


# =========================================
# PURCHASE DISTRIBUTION
# =========================================

print("Mean Purchase Amount:", df['Purchase'].mean())

print("Median Purchase Amount:", df['Purchase'].median())


# =========================================
# GENDER-WISE PURCHASE ANALYSIS
# =========================================

male_purchase = df[df['Gender'] == 'M']['Purchase']

female_purchase = df[df['Gender'] == 'F']['Purchase']


print("Average Purchase Amount (Male):", male_purchase.mean())

print("Average Purchase Amount (Female):", female_purchase.mean())


# =========================================
# PURCHASE DISTRIBUTION VISUALIZATION
# =========================================

sns.histplot(
    male_purchase,
    kde=True,
    color='blue',
    label='Male',
    alpha=0.6
)

sns.histplot(
    female_purchase,
    kde=True,
    color='pink',
    label='Female',
    alpha=0.6
)

plt.legend()

plt.title("Distribution of Purchase Amount by Gender")

plt.show()

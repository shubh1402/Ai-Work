# =========================================
# DELHIVERY FEATURE ENGINEERING PROJECT
# =========================================

# =========================================
# IMPORT REQUIRED LIBRARIES
# =========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("../datasets/delhivery_data.csv")


# =========================================
# BASIC DATA EXPLORATION
# =========================================

print("\n================ DATASET INFO ================\n")

print(df.info())


print("\n================ FIRST 5 ROWS ================\n")

print(df.head())


print("\n================ LAST 5 ROWS ================\n")

print(df.tail())


print("\n================ DATASET SHAPE ================\n")

print(df.shape)


print("\n================ COLUMN NAMES ================\n")

print(df.columns)


print("\n================ STATISTICAL SUMMARY ================\n")

print(df.describe())


# =========================================
# MISSING VALUE ANALYSIS
# =========================================

print("\n================ MISSING VALUES ================\n")

print(df.isnull().sum())


missing_percentage = (
    df.isnull().sum() / len(df)
) * 100


print("\n================ MISSING VALUE PERCENTAGE ================\n")

print(
    missing_percentage.sort_values(
        ascending=False
    )
)


# =========================================
# DUPLICATE VALUE ANALYSIS
# =========================================

print("\n================ DUPLICATE RECORDS ================\n")

print(df.duplicated().sum())


# =========================================
# REMOVE DUPLICATES
# =========================================

df = df.drop_duplicates()

print("\n================ SHAPE AFTER REMOVING DUPLICATES ================\n")

print(df.shape)


# =========================================
# DATETIME FEATURE ENGINEERING
# =========================================

print("\n================ DATETIME CONVERSION ================\n")


datetime_columns = [
    'trip_creation_time',
    'od_start_time',
    'od_end_time'
]


for col in datetime_columns:
    
    df[col] = pd.to_datetime(
        df[col],
        errors='coerce'
    )


print(df[datetime_columns].head())


# =========================================
# EXTRACT DATE FEATURES
# =========================================

print("\n================ DATE FEATURE ENGINEERING ================\n")


df['trip_year'] = df['trip_creation_time'].dt.year

df['trip_month'] = df['trip_creation_time'].dt.month

df['trip_day'] = df['trip_creation_time'].dt.day

df['trip_hour'] = df['trip_creation_time'].dt.hour

df['trip_weekday'] = df['trip_creation_time'].dt.day_name()


print(
    df[
        [
            'trip_year',
            'trip_month',
            'trip_day',
            'trip_hour',
            'trip_weekday'
        ]
    ].head()
)


# =========================================
# DELIVERY TIME FEATURE ENGINEERING
# =========================================

print("\n================ DELIVERY TIME ENGINEERING ================\n")


df['actual_delivery_time'] = (
    df['od_end_time'] - df['od_start_time']
).dt.total_seconds() / 3600


print(df['actual_delivery_time'].head())


# =========================================
# KPI FEATURE ENGINEERING
# =========================================

print("\n================ KPI FEATURE ENGINEERING ================\n")


# Delivery Efficiency Ratio

df['delivery_efficiency'] = (
    df['actual_time'] / df['osrm_time']
)


# Distance Efficiency Ratio

df['distance_efficiency'] = (
    df['actual_distance_to_destination']
    /
    df['osrm_distance']
)


print(
    df[
        [
            'delivery_efficiency',
            'distance_efficiency'
        ]
    ].head()
)


# =========================================
# OUTLIER ANALYSIS
# =========================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df['actual_time']
)

plt.title("Actual Delivery Time Outlier Analysis")

plt.savefig(
    "../screenshots/Actual_Delivery_Time_Outlier.png"
)

plt.show()


plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df['actual_distance_to_destination']
)

plt.title("Actual Distance Outlier Analysis")

plt.savefig(
    "../screenshots/Actual_Distance_Outlier.png"
)

plt.show()


# =========================================
# MONTHLY SHIPMENT ANALYSIS
# =========================================

monthly_orders = (
    df.groupby('trip_month')
    ['data']
    .count()
)


plt.figure(figsize=(10, 5))

monthly_orders.plot(
    kind='bar'
)

plt.title("Monthly Shipment Analysis")

plt.xlabel("Month")

plt.ylabel("Shipment Count")

plt.savefig(
    "../screenshots/Monthly_Shipment_Analysis.png"
)

plt.show()


# =========================================
# WEEKDAY SHIPMENT ANALYSIS
# =========================================

weekday_orders = (
    df.groupby('trip_weekday')
    ['data']
    .count()
)


plt.figure(figsize=(10, 5))

weekday_orders.plot(
    kind='bar'
)

plt.title("Weekday Shipment Analysis")

plt.xlabel("Weekday")

plt.ylabel("Shipment Count")

plt.savefig(
    "../screenshots/Weekday_Shipment_Analysis.png"
)

plt.show()


# =========================================
# DELIVERY EFFICIENCY DISTRIBUTION
# =========================================

plt.figure(figsize=(10, 5))

sns.histplot(
    df['delivery_efficiency'],
    bins=50,
    kde=True
)

plt.title("Delivery Efficiency Distribution")

plt.savefig(
    "../screenshots/Delivery_Efficiency_Distribution.png"
)

plt.show()


# =========================================
# CORRELATION ANALYSIS
# =========================================

numerical_columns = df.select_dtypes(
    include=np.number
).columns


correlation_matrix = (
    df[numerical_columns]
    .corr()
)


plt.figure(figsize=(15, 10))

sns.heatmap(
    correlation_matrix,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig(
    "../screenshots/Correlation_Heatmap.png"
)

plt.show()


# =========================================
# TOP SOURCE CITIES
# =========================================

top_source_cities = (
    df['source_name']
    .value_counts()
    .head(10)
)


plt.figure(figsize=(12, 6))

top_source_cities.plot(
    kind='bar'
)

plt.title("Top Source Cities")

plt.xlabel("City")

plt.ylabel("Shipment Count")

plt.savefig(
    "../screenshots/Top_Source_Cities.png"
)

plt.show()


# =========================================
# TOP DESTINATION CITIES
# =========================================

top_destination_cities = (
    df['destination_name']
    .value_counts()
    .head(10)
)


plt.figure(figsize=(12, 6))

top_destination_cities.plot(
    kind='bar'
)

plt.title("Top Destination Cities")

plt.xlabel("City")

plt.ylabel("Shipment Count")

plt.savefig(
    "../screenshots/Top_Destination_Cities.png"
)

plt.show()


# =========================================
# BUSINESS INSIGHTS
# =========================================

print("\n================ BUSINESS INSIGHTS ================\n")

print("""
1. Delivery efficiency varies significantly across shipments.

2. Certain routes contribute heavily to operational load.

3. Monthly shipment trends indicate seasonal logistics demand.

4. Delivery delays can be analyzed using engineered KPI metrics.

5. Operational inefficiencies are visible through distance and time variance.

6. Feature engineering improves downstream analytics readiness.
""")


# =========================================
# EXPORT CLEANED DATASET
# =========================================

df.to_csv(
    "../results/cleaned_delhivery_data.csv",
    index=False
)


print("\n================ PIPELINE EXECUTED SUCCESSFULLY ================\n")
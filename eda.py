import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== PART A: PROFILING, CLEANING & EDA STORY ===")

# Task 1: Load dataset ONCE and save offline fallback
try:
    df = sns.load_dataset('titanic')
    print("Loaded 'titanic' dataset via Seaborn.")
except Exception as e:
    print(f"Network load failed: {e}. Falling back to local 'titanic.csv'.")
    df = pd.read_csv('titanic.csv')

# Save committed offline fallback immediately
df.to_csv("titanic.csv", index=False)
print("Saved offline fallback to 'analytics/titanic.csv'.")

# Profile dataset
print("\n--- Dataset Profile ---")
print(f"Shape: {df.shape}")
print("\nDataFrame Info:")
df.info()
print("\nDataFrame Summary Statistics:")
print(df.describe(include='all'))

# Missing values calculation
print("\n--- Missing Value Percentages ---")
missing_counts = df.isnull().sum()
missing_pcts = (missing_counts / len(df)) * 100
missing_info = pd.DataFrame({'Missing_Count': missing_counts, 'Missing_Percentage': missing_pcts})
print(missing_info[missing_info['Missing_Count'] > 0])

# Task 2: Missing Value Handling based on Threshold Rules
# Rule: <5% -> drop rows; 5%-30% -> impute; >30% -> drop column or encode 'missing'
df_clean = df.copy()

# 'embarked' & 'embark_town': ~0.22% missing (<5%) -> Drop rows
df_clean = df_clean.dropna(subset=['embarked', 'embark_town'])

# 'age': ~19.87% missing (5%-30%) -> Impute with median
age_median = df_clean['age'].median()
df_clean['age'] = df_clean['age'].fillna(age_median)

# 'deck': ~77.10% missing (>30%) -> Encode "Missing" as its own category to preserve info
df_clean['deck'] = df_clean['deck'].cat.add_categories('Missing').fillna('Missing') if hasattr(df_clean['deck'], 'cat') else df_clean['deck'].fillna('Missing')

print("\nMissing values remaining after cleaning:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# Save cleaned dataset for modeling phase
df_clean.to_csv("titanic_cleaned.csv", index=False)

# Task 3: Univariate Analysis (Age & Fare)
print("\n--- Univariate Analysis ---")

for col in ['age', 'fare']:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    print(f"[{col.upper()}] Outliers count (IQR Rule): {len(outliers)}")

# Fare Skewness Check
fare_mean = df_clean['fare'].mean()
fare_median = df_clean['fare'].median()
fare_mode = df_clean['fare'].mode()[0]
print(f"Fare Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")
# Since Mean (32.20) > Median (14.45) > Mode (8.05), Fare is strongly right-skewed.

# Task 4: Bivariate Analysis & Correlation Matrix
print("\n--- Bivariate Analysis ---")
# Survival rates by sex
print("Survival rate by Sex:")
print(df_clean.groupby('sex')['survived'].mean())

# Survival rates by Pclass
print("\nSurvival rate by Pclass:")
print(df_clean.groupby('pclass')['survived'].mean())

# Survival rates by Sex and Pclass combined
print("\nSurvival rate by Sex and Pclass:")
print(df_clean.groupby(['sex', 'pclass'])['survived'].mean())

# Correlation Matrix (strictly the 6 numeric columns, excluding adult_male and alone)
numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df_clean[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (6 Core Numeric Features)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("\nCorrelation matrix saved to 'correlation_heatmap.png'.")

# Identify top 2 off-diagonal absolute correlations
corr_pairs = corr_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0].sort_values(ascending=False).drop_duplicates()
print("\nTop 2 strongest off-diagonal correlations:")
print(corr_pairs.head(2))

# Task 5: Multivariate Data Story (4 Charts)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Survival Rate by Sex and Pclass
sns.barplot(data=df_clean, x='pclass', y='survived', hue='sex', ax=axes[0, 0])
axes[0, 0].set_title('Chart 1: Survival Rate by Class & Sex')

# Chart 2: Fare Distribution by Pclass & Survival Status
sns.boxplot(data=df_clean, x='pclass', y='fare', hue='survived', ax=axes[0, 1])
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('Chart 2: Log Fare vs Pclass & Survival')

# Chart 3: Age Distribution vs Survival Status
sns.kdeplot(data=df_clean, x='age', hue='survived', common_norm=False, ax=axes[1, 0])
axes[1, 0].set_title('Chart 3: Age Density by Survival Status')

# Chart 4: Family Size (SibSp + Parch) vs Survival
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
sns.lineplot(data=df_clean, x='family_size', y='survived', marker='o', ax=axes[1, 1])
axes[1, 1].set_title('Chart 4: Survival Rate vs Family Size')

plt.tight_layout()
plt.savefig('eda_charts.png')
print("Multivariate charts saved to 'eda_charts.png'.")

# Task 6: Z-score Standardization Sanity Check (EDA level only)
df_std = df_clean[['age', 'fare']].copy()
df_std['age_z'] = (df_std['age'] - df_std['age'].mean()) / df_std['age'].std()
df_std['fare_z'] = (df_std['fare'] - df_std['fare'].mean()) / df_std['fare'].std()

print("\n--- Standardization Sanity Check ---")
print(f"Age  - Mean: {df_std['age_z'].mean():.4f}, Std: {df_std['age_z'].std():.4f}")
print(f"Fare - Mean: {df_std['fare_z'].mean():.4f}, Std: {df_std['fare_z'].std():.4f}")
print("Confirming transformed features have approximately mean 0 and std 1.")
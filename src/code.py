import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

# Analisi del dataset e Data Preparation

df = pd.read_csv('../student_depression.csv')

# Target
target = 'Depression'

# Rimuovo colonne non informative
columns_to_drop = ['id','City']
df = df.drop(columns=columns_to_drop, errors='ignore')


# Controllo valori null
print("\nVerifica valori null nel dataset:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])


# Feature numeriche vs target
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target in numerical_cols:
    numerical_cols.remove(target)

numerical_corr = {}
numerical_anomaly = {}

for col in numerical_cols:
    col_data = df[col].dropna()
    # Correlazione con la target
    corr = col_data.corr(df[target])
    numerical_corr[col] = corr

    # Statistiche distribuzione
    mean = col_data.mean()
    std = col_data.std()
    skewness = skew(col_data)
    kurt = kurtosis(col_data)
    # Flag anomalie basate sulla distribuzione
    flag_dist = abs(skewness) > 1 or kurt > 5 or std < 0.01
    numerical_anomaly[col] = flag_dist

# Ordina feature numeriche per correlazione assoluta
sorted_corr = dict(sorted(numerical_corr.items(), key=lambda x: abs(x[1]), reverse=True))

print("Feature numeriche più correlate alla target 'Depression':")
for col, corr in sorted_corr.items():
    print(f"{col}: {corr:.3f}")

# Feature categoriche vs target
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_dependence = {}
categorical_anomaly = {}

for col in categorical_cols:
    col_data = df[col].dropna()
    # Percentuale differenza massima tra le classi della target
    ct = pd.crosstab(col_data, df[target], normalize='index')
    max_diff = abs(ct[1] - ct[0]).max()
    categorical_dependence[col] = max_diff

    # Distribuzione feature
    n_unique = col_data.nunique()
    value_counts = col_data.value_counts(normalize=True) * 100
    flag_dist = n_unique <= 1 or value_counts.max() > 90
    categorical_anomaly[col] = flag_dist

# Ordina feature categoriche per dipendenza
sorted_dep = dict(sorted(categorical_dependence.items(), key=lambda x: x[1], reverse=True))

print("\nFeature categoriche con maggiore dipendenza dalla target 'Depression':")
for col, dep in sorted_dep.items():
    print(f"{col}: {dep:.3f}")


print("\nFeature numeriche anomale candidate per rimozione (distribuzione o bassa correlazione):")
numerical_candidates = [col for col in numerical_cols if numerical_anomaly[col] or abs(numerical_corr[col]) < 0.05]
print(numerical_candidates)

print("\nFeature categoriche anomale candidate per rimozione (distribuzione o bassa dipendenza):")
categorical_candidates = [col for col in categorical_cols if categorical_anomaly[col] or categorical_dependence[col] < 0.05]
print(categorical_candidates)

# Rimuovo colonne non informative
columns_to_drop = ['Work Pressure', 'CGPA', 'Job Satisfaction','Profession']
df = df.drop(columns=columns_to_drop, errors='ignore')

print("\nDataset dopo Data Cleaning:")
print(df.head())
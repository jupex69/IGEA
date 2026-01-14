import pandas as pd

# =================================================================
# CONFIGURAZIONE INIZIALE
# include il data Cleaning
# =================================================================
DATASET_PATH = '../student_depression.csv'
target = 'Depression'

print("--- FASE 1: CARICAMENTO & CLEANING (Da Common) ---")

# 1. Carichiamo il dataset grezzo
df_raw = pd.read_csv(DATASET_PATH)

# 2. Importiamo la funzione di pulizia dal file 'common.py'
from common import getCleanedData

print("Richiesta dataset pulito...")
# Passiamo il dataframe grezzo alla funzione comune
df = getCleanedData(df_raw)

print(f"Dataset ricevuto! Dimensioni: {df.shape}")
print(df.head())


print("\n--- FASE 2: TRASFORMAZIONE DEGREE ---")

def map_degree(deg):
    if deg == "'Class 12'":
        return 'Diploma'
    elif deg.startswith('B'):
        return 'Titolo_primo_livello'
    elif deg.startswith('M') or deg in ['PhD', 'MD']:
        return 'Titolo_secondo_livello'
    else:
        return 'Titolo_secondo_livello'  # categoria residuale

df['Degree_level'] = df['Degree'].apply(map_degree)

print("Distribuzione Degree_level:")
print(df['Degree_level'].value_counts())

print("\n--- FASE 3: TRASFORMAZIONE CGPA ---")

df['CGPA_30'] = 2.4 * df['CGPA'] + 6


print("Statistiche CGPA_30:")
print(df['CGPA_30'].describe())



print("\n--- FASE 4: FEATURE SELECTION ---")

columns_to_drop = [
    'City',
    'Work Pressure',
    'Job Satisfaction',
    'Profession',
    'Degree',
    'CGPA'
]

df = df.drop(columns=columns_to_drop, errors='ignore')

print(f"Feature rimosse: {columns_to_drop}")
print(f"Dimensioni dopo Feature Selection: {df.shape}")



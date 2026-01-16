# pipeline2.py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# =================================================================
# CONFIGURAZIONE
# =================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "student_depression.csv"

TARGET = 'Depression'

# =================================================================
# FASE 1: CARICAMENTO & CLEANING
# =================================================================
print("--- PIPELINE 2 | DATA PREPARATION ---")
print("--- FASE 1: CARICAMENTO & CLEANING ---")

df_raw = pd.read_csv(DATASET_PATH)

from data_cleaning import getCleanedData
print("Richiesta dataset pulito...")
df = getCleanedData(df_raw)

print(f"Dataset ricevuto! Dimensioni: {df.shape}")
print(df.head())

# =================================================================
# FASE 2: FEATURE ENGINEERING – DEGREE
# =================================================================
print("\n--- FASE 2: TRASFORMAZIONE DEGREE ---")

def map_degree(deg):
    if deg == "'Class 12'":
        return 'Diploma'
    elif isinstance(deg, str) and deg.startswith('B'):
        return 'Titolo_primo_livello'
    elif isinstance(deg, str) and (deg.startswith('M') or deg in ['PhD', 'MD']):
        return 'Titolo_secondo_livello'
    else:
        return 'Titolo_secondo_livello'

df['Degree_level'] = df['Degree'].apply(map_degree)

print("Distribuzione Degree_level:")
print(df['Degree_level'].value_counts())

# =================================================================
# FASE 3: FEATURE ENGINEERING – CGPA
# =================================================================
print("\n--- FASE 3: TRASFORMAZIONE CGPA ---")

df['CGPA_30'] = 2.4 * df['CGPA'] + 6

print("Statistiche CGPA_30:")
print(df['CGPA_30'].describe())

# =================================================================
# FASE 4: FEATURE SELECTION
# =================================================================
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

# =================================================================
# FASE 5: SPLIT X / y (RAW)
# =================================================================
print("\n--- FASE 5: Split X / y (RAW) ---")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =================================================================
# FASE 6: DEFINIZIONE PREPROCESSOR (NON FITTATO)
# =================================================================
print("\n--- FASE 6: DEFINIZIONE PREPROCESSING ---")

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Feature categoriche:", categorical_features)
print("Feature numeriche:", numeric_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        ), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='drop'
)

print("Pipeline 2 pronta.")

# =================================================================
# EXPORT
# =================================================================
__all__ = ['X', 'y', 'preprocessor']

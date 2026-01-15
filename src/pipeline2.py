import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =================================================================
# CONFIGURAZIONE INIZIALE
# include il data Cleaning
# =================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "student_depression.csv"

target = 'Depression'

print("--- FASE 1: CARICAMENTO & CLEANING ---")

# 1. Carichiamo il dataset grezzo
df_raw = pd.read_csv(DATASET_PATH)

# 2. Importiamo la funzione di pulizia dal file 'data_understanding.py'
from data_cleaning import getCleanedData

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


# =================================================================
# FASE 5: PREPROCESSING (Scaling e Encoding)
# =================================================================

print("\n--- FASE 5: PREPROCESSING ---")

# Separiamo le feature numeriche e categoriche
X = df.drop(columns=[target])
y = df[target]

# Identifichiamo le variabili categoriche e numeriche
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing: One-Hot Encoding per le variabili categoriche e StandardScaler per quelle numeriche
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='drop'
)

# Creiamo una pipeline di preprocessing (questa parte è per preparare i dati da usare nel modeling)
preprocessing_pipeline = Pipeline(steps=[('preprocess', preprocess)])

# Applichiamo la pipeline di preprocessing sui dati
X_processed = preprocessing_pipeline.fit_transform(X)

print("Trasformazione completata! Il dataset è ora pronto per essere utilizzato nel modello.")
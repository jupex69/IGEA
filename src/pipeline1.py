import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "student_depression.csv"
TARGET = 'Depression'

print("--- PIPELINE 1 | DATA PREPARATION ---")
# ==============================================================================
# FASE 1: CARICAMENTO & CLEANING
# ==============================================================================
print("--- FASE 1: CARICAMENTO & CLEANING")
try:
    df_raw = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    df_raw = pd.read_csv('student_depression_dataset.csv')

from data_cleaning import getCleanedData
print("Richiesta dataset pulito...")
df = getCleanedData(df_raw)
print(f"Dataet Ricevuto!: {df.shape}")
print(df.head())

print("\n--- FASE 2: TRASFORMAZIONE AGE ---")

def map_age_custom(age):
    if age < 22:
        return '18-21'
    elif age < 26:
        return '22-25'
    elif age < 30:
        return '26-29'
    else:
        return '30+'

# Applicazione della trasformazione
df['Age_group'] = df['Age'].apply(map_age_custom)


# ==============================================================================
# FASE 3: FEATURE ENGINEERING
# ==============================================================================
print("\n--- FASE 3: FEATURE ENGINEERING ---")
# 1. Pulizia preventiva numerica
cols_to_fix = ['Sleep Duration', 'Academic Pressure', 'Financial Stress']
for col in cols_to_fix:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 2. Creazione Feature
required_cols = {'Sleep Duration', 'Academic Pressure', 'Financial Stress'}
if required_cols.issubset(df.columns):
    # Debito di sonno
    df['Sleep_Debt'] = (8 - df['Sleep Duration']).clip(lower=0)

    # Stress amplificato
    pressione = df['Academic Pressure'] + df['Financial Stress']
    df['Stress_Amplified'] = pressione * (1 + df['Sleep_Debt'])
    print("✅ Feature 'Stress_Amplified' creata con successo.")
else:
    print(f"⚠️ Attenzione impossibile creare 'Stress_Amplified'. Mancano colonne  base: {required_cols}.")

# ==============================================================================
# FASE 4: FEATURE SELECTION
# ==============================================================================
print("\n---FASE 4:FEATURE SELECTION---")

cols_noise = ['Profession', 'CGPA', 'Job Satisfaction', 'Work Pressure']
cols_redundant = ['Academic Pressure', 'Financial Stress', 'Sleep Duration', 'Sleep_Debt', 'Age']
cols_low_info = ['Degree', 'City']

cols_to_drop = cols_noise + cols_redundant + cols_low_info
print(f"📉 Rimozione manuale feature: {cols_to_drop}")
df = df.drop(columns=cols_to_drop, errors='ignore')

# ==============================================================================
# FASE 5: SPLIT X / y
# ==============================================================================
# Resettiamo l'indice ORA per entrambi, così siamo sicuri che
# la riga 0 di X corrisponda alla riga 0 di y.
df = df.reset_index(drop=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Dimensioni finali X: {X.shape}")

# ==============================================================================
# FASE 6: DEFINIZIONE PREPROCESSOR
# ==============================================================================
print("\n---FASE 6: DEFINIZIONE PREPROCESSOR---")
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='drop'
)

print("✅ Pipeline 1 pronta per l'export.")
__all__ = ['X', 'y', 'preprocessor']

# ==============================================================================
# VISUALIZZAZIONE CORRETTA (DEBUG)
# ==============================================================================
if __name__ == "__main__":
    print("\n--- ANALISI CORRELAZIONE (Pipeline 1 Fixed) ---")

    # 1. Ricostruiamo il dataframe unendo X e y (Garantito allineamento)
    df_viz = pd.concat([X, y], axis=1)

    # 2. Applichiamo One Hot Encoding temporaneo (come nella Pipeline 2)
    #    Questo "esplode" le categorie (es. Dietary Habits -> Dietary Habits_Unhealthy)
    #    permettendoci di vedere correlazioni specifiche.
    df_viz_encoded = pd.get_dummies(df_viz, drop_first=True, dtype=float)

    # 3. Calcolo Correlazione
    correlations = df_viz_encoded.corr()[TARGET].drop(TARGET)

    # 4. Top 5 Assolute
    top_5 = correlations.abs().sort_values(ascending=False).head(5)

    # Recuperiamo i valori col segno originale
    final_top_5 = correlations[top_5.index]

    print("🏆 Top 5 Feature (Valori Reali):")
    print(final_top_5)

    # Grafico
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=final_top_5.values, y=final_top_5.index, hue=final_top_5.index, legend=False, palette='viridis')
        plt.title('Top 5 Features Pipeline 1', fontsize=15)
        plt.xlabel('Correlazione Pearson')
        plt.grid(axis='x', alpha=0.3)
        plt.show()
    except Exception:
        pass
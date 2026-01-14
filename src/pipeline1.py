import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Se i file sono nella stessa cartella, questo aiuta l'import
sys.path.append('.')

# ==============================================================================
# 0. CONFIGURAZIONE & IMPORT
# ==============================================================================
DATASET_PATH = '../student_depression.csv'
target = 'Depression'

# ==============================================================================
# FASE 1: DATA CLEANING & PREPARATION (Modulo Esterno)
# ==============================================================================
print("--- FASE 1: CARICAMENTO & CLEANING ---")

# 1. Caricamento Dataset Grezzo
df_raw = pd.read_csv(DATASET_PATH)

# 2. Importazione ed Esecuzione Pipeline di Pulizia (da common.py)
#    Questa funzione applica: drop ID/City, rimozione statistica outlier, ecc.
from common import getCleanedData
print("Richiesta dataset pulito...")

df = getCleanedData(df_raw)

print(f"Dataset ricevuto! Dimensioni: {df.shape}")
print(df.head())

# ==============================================================================
# FASE 2: FEATURE ENGINEERING (Pipeline A - Arricchimento)
# ==============================================================================
print("\n--- FASE 2: FEATURE ENGINEERING & ENCODING ---")

# 1. Encoding delle variabili categoriche
#    Trasformiamo tutte le stringhe in numeri per poter fare calcoli matematici.
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

# 2. Costruzione delle Nuove Feature (Construction)
#    Verifichiamo che le colonne necessarie ("Ingredienti") siano sopravvissute alla Fase 1.
required_cols = {'Sleep Duration', 'Academic Pressure', 'Financial Stress'}

if required_cols.issubset(df.columns):
    # A. Calcolo del Debito di Sonno (Quanto dormi meno di 8 ore)
    df['Sleep_Debt'] = (8 - df['Sleep Duration']).clip(lower=0)

    # B. Calcolo Stress Amplificato (La tua formula personalizzata)
    #    Ipotesi: Lo stress esterno pesa di più se sei stanco.
    pressione_esterna = df['Academic Pressure'] + df['Financial Stress']
    df['Stress_Amplified'] = pressione_esterna * (1 + df['Sleep_Debt'])

    print("✅ Feature 'Stress_Amplified' creata con successo.")
else:
    print(f"⚠️ Attenzione: Impossibile creare Stress_Amplified. Mancano colonne base: {required_cols}")

# ==============================================================================
# FASE 3: FEATURE SELECTION & PRE-PROCESSING
# ==============================================================================
print("\n--- FASE 3: SELEZIONE, SCALING E ANALISI ---")

# --- 3.1 SELEZIONE MANUALE (Rimozione Fisica) ---
# Eliminiamo le colonne che non vogliamo dare in pasto al modello.

# A) Rimozione "Rumore" (Feature poco informative per il contesto)
cols_noise = ['Profession', 'CGPA', 'Job Satisfaction', 'Work Pressure']

# B) Rimozione "Ridondanze" (Genitori di Stress_Amplified)
#    Se teniamo sia la Madre (Academic Pressure) che la Figlia (Stress_Amplified),
#    creiamo multicollinearità. Rimuoviamo le originali per forzare l'uso della nuova feature.
cols_redundant = []
if 'Stress_Amplified' in df.columns:
    cols_redundant = ['Academic Pressure', 'Financial Stress', 'Sleep Duration', 'Sleep_Debt']

# C) Esecuzione rimozione feature non relative al contesto italiano
cols_lowInformation = ['id', 'Degree', 'City']

cols_to_drop = cols_noise + cols_redundant + cols_lowInformation
print(f"📉 Rimozione manuale feature: {cols_to_drop}")
df = df.drop(columns=cols_to_drop, errors='ignore')

# --- 3.2 DEFINIZIONE MATRICE FEATURE (X) E TARGET (y) ---
X = df.drop(columns=[target])
y = df[target]
print(f"Dimensioni finali X: {X.shape}")

# --- 3.3 FEATURE SCALING (Standardizzazione) ---
# Necessario per algoritmi come Logistic Regression (Gradient Descent).
# Trasforma i dati per avere Media=0 e Deviazione Standard=1.
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- 3.4 VALIDAZIONE (Analisi Correlazione Pearson) ---
# Verifichiamo quali sono le feature più potenti rimaste nel dataset finale.

final_corr = X_scaled.copy()
final_corr[target] = y.reset_index(drop=True)

# Calcolo classifica (valore assoluto)
ranking = final_corr.corr()[target].abs().sort_values(ascending=False).drop(target)
top_5_features = ranking.head(5)

print("\n🏆 CLASSIFICA FINALE DELLE FEATURE (Top 5):")
for i, (feat, val) in enumerate(top_5_features.items(), 1):
    print(f"{i}. {feat}: {val:.4f}")

# Visualizzazione Grafica
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_5_features.values,
    y=top_5_features.index,
    hue=top_5_features.index,
    palette='rocket',
    legend=False
)
plt.title("Top 5 Feature Selezionate (Pipeline A - Engineered)")
plt.xlabel("Coefficiente di Pearson")
plt.tight_layout()
plt.show()

print(df.head())
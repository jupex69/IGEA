import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =================================================================
# CONFIGURAZIONE INIZIALE
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

# =================================================================
# FASE 2: FEATURE ENGINEERING (Specifico Pipeline A - Giuseppe)
# =================================================================
print("\n--- FASE 2: FEATURE ENGINEERING & ENCODING ---")

# Encoding necessario per poter fare calcoli matematici sulle colonne object
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

# Creazione delle Feature "Stress_Amplified"
# Verifichiamo che le colonne necessarie esistano ancora dopo la pulizia di common
required_cols = {'Sleep Duration', 'Academic Pressure', 'Financial Stress'}

if required_cols.issubset(df.columns):
    # 1. Creazione Debito di Sonno (8 ore meno le ore dormite)
    df['Sleep_Debt'] = (8 - df['Sleep Duration']).clip(lower=0)

    # 2. Creazione Stress Amplificato
    pressione_esterna = df['Academic Pressure'] + df['Financial Stress']
    df['Stress_Amplified'] = pressione_esterna * (1 + df['Sleep_Debt'])

    print("✅ Feature 'Stress_Amplified' creata con successo.")
else:
    print(f"⚠️ Attenzione: Mancano le colonne {required_cols} per creare Stress_Amplified.")

# =================================================================
# FASE 3: SCALING & SELECTION
# =================================================================
print("\n--- FASE 3: SCALING & SELECTION ---")

# Ora usiamo la variabile 'target' definita all'inizio
X = df.drop(columns=[target])
y = df[target]

# Feature Scaling (StandardScaler)
# Importante: Scaliamo le X per rendere confrontabili Stress (es. 50) e Sonno (es. 4)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Selezione finale con Pearson
final_corr = X_scaled.copy()
final_corr[target] = y.reset_index(drop=True)

# Calcoliamo la correlazione assoluta e ordiniamo
correlations = final_corr.corr()[target].abs().sort_values(ascending=False).drop(target)

# GESTIONE DELLA RIDONDANZA
# Se la nostra feature creata è potente, rimuoviamo i "genitori" per evitare doppioni
if 'Stress_Amplified' in correlations.head(5).index:
    cols_to_remove = ['Academic Pressure', 'Financial Stress', 'Sleep Duration', 'Sleep_Debt']
    correlations = correlations.drop(labels=cols_to_remove, errors='ignore')
    print("✅ Rimosse feature ridondanti (Genitori di Stress_Amplified).")

top_5_features = correlations.head(5)

print("\n🏆 CLASSIFICA FINALE DELLE FEATURE (Top 5):")
for i, (feat, val) in enumerate(top_5_features.items(), 1):
    print(f"{i}. {feat}: {val:.4f}")

# Visualizzazione
plt.figure(figsize=(10, 6))
sns.barplot(x=top_5_features.values, y=top_5_features.index, hue=top_5_features.index, palette='rocket', legend=False)
plt.title("Top 5 Feature Selezionate (Pipeline A)")
plt.xlabel("Coefficiente di Pearson")
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configurazione
DATASET_PATH = '../student_depression.csv'
print("--- FASE 1: DATA CLEANING & STATISTICAL ANALYSIS ---")
df = pd.read_csv(DATASET_PATH)
target = 'Depression'
# =================================================================
# FASE 1: DATA SELECTION STATISTICA (Il lavoro del tuo collega)
# =================================================================


# Pulizia Base
columns_to_drop_manual = ['id', 'City', 'Have you ever had suicidal thoughts ?']
df = df.drop(columns=columns_to_drop_manual, errors='ignore')


# =================================================================
# FASE 2: FEATURE ENGINEERING Giuseppe (Il Tuo Lavoro - La scelta vincente)
# =================================================================
print("\n--- FASE 2: FEATURE ENGINEERING & ENCODING ---")

# Encoding necessario per i calcoli
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

# Assicuriamoci che le colonne chiave esistano ancora
if {'Sleep Duration', 'Academic Pressure', 'Financial Stress'}.issubset(df.columns):
    # Creazione Debito di Sonno (Logica continua, superiore al binning)
    df['Sleep_Debt'] = (8 - df['Sleep Duration']).clip(lower=0)

    # Creazione Stress Amplificato (La tua formula vincente)
    pressione_esterna = df['Academic Pressure'] + df['Financial Stress']
    df['Stress_Amplified'] = pressione_esterna * (1 + df['Sleep_Debt'])
    print("✅ Feature 'Stress_Amplified' creata con successo.")
else:
    print("⚠️ Alcune colonne necessarie sono state rimosse nella Fase 1.")

# =================================================================
# FASE 3: SCALING & SELECTION (Integrazione Finale)
# =================================================================
print("\n--- FASE 3: SCALING & SELECTION ---")

X = df.drop(columns=[target])
y = df[target]

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Selezione finale con Pearson
final_corr = X_scaled.copy()
final_corr[target] = y.reset_index(drop=True)
correlations = final_corr.corr()[target].abs().sort_values(ascending=False).drop(target)

# Gestione della ridondanza (Teniamo solo la migliore)
if 'Stress_Amplified' in correlations.head(5).index:
    #Rimuoviamo le colonne che generano Stress_Amplified per evitare multicollinearità
    correlations = correlations.drop(['Academic Pressure', 'Financial Stress', 'Sleep Duration', 'Sleep_Debt'], errors='ignore')

top_features = correlations.head(5)

print("\n🏆 CLASSIFICA FINALE DELLE FEATURE (Top 5):")
for i, (feat, val) in enumerate(top_features.items(), 1):
    print(f"{i}. {feat}: {val:.4f}")

# Visualizzazione
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index, palette='rocket', legend=False)
plt.title("Top Feature Selezionate (Dopo Cleaning Statistico e Feature Engineering)")
plt.xlabel("Coefficiente di Pearson")
plt.show()
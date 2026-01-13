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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

# ==================
#   DATA CLEANING
# ==================
def getCleanedData(df_input):
    """
    Esegue la logica di cleaning definita:
    - Rimozione colonne manuali
    - Rimozione Null
    - Rimozione Duplicati
    - Rimozione Outlier (IQR 0.30 - 0.70)
    """
    # Lavoriamo su una copia
    df = df_input.copy()
    target = 'Depression'

    print("\n=== DATA CLEANING ===")

    # Dimensioni iniziali del dataset
    print("\nDimensioni iniziali del dataset:", df.shape)

    # Rimuovo colonne non informative
    columns_to_drop = ['id', 'City', 'Have you ever had suicidal thoughts ?']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    print(f"Totale valori nulli nel dataset: {df.isnull().sum().sum()}")

    # Rimuovo i null (aggiunto per coerenza con il calcolo successivo)
    df = df.dropna()

    # Controllo duplicati
    num_duplicates = df.duplicated().sum()
    print(f"Numero di righe duplicate: {num_duplicates}")
    # Nota: il codice originale contava i duplicati ma non li rimuoveva esplicitamente col drop_duplicates.
    # Rispetto la logica originale e non aggiungo df.drop_duplicates() se non c'era.

    # Controllo coerenza target (es. binario)
    if target in df.columns:
        print("\nValori unici del target:", df[target].unique())

    # Controllo valori anomali (outlier) per colonne numeriche
    print("\nControllo outlier (IQR 0.30-0.70) per colonne numeriche:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Escludiamo il target dalla rimozione outlier se è numerico (binario)
    if target in numeric_cols:
        numeric_cols = numeric_cols.drop(target)

    # Salviamo la lunghezza iniziale per il calcolo delle righe rimosse
    len_before_outliers = len(df)


    righe_iniziali = len(df)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.30)
        Q3 = df[col].quantile(0.70)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        """ STAMPA OUTLIER
        print(f"\nOutlier per la colonna '{col}': {outliers.shape[0]}")

        if not outliers.empty:
            print(outliers[[col]])
        """

        # Rimozione outlier SOLO per quella colonna
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    righe_finali = len(df)
    print(f"Righe rimosse complessivamente: {righe_iniziali - righe_finali}")

    print("\nControllo valori negativi nelle colonne numeriche:")
    colonne_con_negativi = []

    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"{col}: contiene valori negativi")
            colonne_con_negativi.append(col)

    if not colonne_con_negativi:
        print("Nessuna colonna contiene valori negativi")

    # Dimensioni finali del dataset
    print("\nDimensioni finali del dataset:", df.shape, "\n")

    return df


# ===================================
#   DATA UNDERSTANDING & EXECUTION
# ===================================
if __name__ == "__main__":

    print("\n=== DATA UNDERSTANDING ===")

    # Caricamento dataset
    # Usa try/except per gestire i percorsi se lanci da cartelle diverse
    try:
        df = pd.read_csv('../student_depression.csv')
    except FileNotFoundError:
        df = pd.read_csv('student_depression.csv')

    # Target
    target = 'Depression'

    # Controllo bilanciamento
    print("\nControllo bilanciamento dei dati")
    print("Numero di elementi per la classe Depressi", len(df[(df['Depression'] == 1)]))
    print("Numero di elementi per la classe Non depressi", len(df[(df['Depression'] == 0)]))

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

    print("\nFeature numeriche più correlate alla target 'Depression':")
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

    numerical_candidates = [col for col in numerical_cols if numerical_anomaly[col] or abs(numerical_corr[col]) < 0.05]
    print("\nFeature numeriche anomale candidate per rimozione (distribuzione o bassa correlazione):", numerical_candidates)

    categorical_candidates = [col for col in categorical_cols if categorical_anomaly[col] or categorical_dependence[col] < 0.05]
    print("\nFeature categoriche anomale candidate per rimozione (distribuzione o bassa dipendenza):", categorical_candidates, "\n")

    if 'Have you ever had suicidal thoughts ?' in df.columns:
        ct = pd.crosstab(
            df['Have you ever had suicidal thoughts ?'],
            df['Depression'],
            normalize='index'
        )
        print(ct)

    # TEST DELLA FUNZIONE
    df_cleaned = getCleanedData(df)
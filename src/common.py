import pandas as pd
from scipy.stats import skew, kurtosis

# ==================
#   DATA CLEANING
# ==================


def getCleanedData(df_input):
    """
    Esegue SOLO l'azione di pulizia sui dati:
    - Rimozione colonne manuali
    - Rimozione Null e Duplicati
    - Rimozione Outlier (IQR 0.30 - 0.70)
    """
    # 1. Copia e parametri locali
    df = df_input.copy()
    target = 'Depression'

    # 2. Rimozione colonne non informative
    columns_to_drop = ['id', 'City', 'Have you ever had suicidal thoughts ?']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # 3. Pulizia righe: Null e Duplicati
    df = df.dropna()
    df = df.drop_duplicates()

    # 4. Rimozione Outlier (IQR 0.30-0.70)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if target in numeric_cols:
        numeric_cols = numeric_cols.drop(target)

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

        # Filtro progressivo
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

 # ========================
 #  DATA UNDERSTANDING
 # ========================
if __name__ == "__main__":

    df = pd.read_csv('../student_depression.csv')

    target = 'Depression'

    # --- ANALISI ---
    print("\n=== DATA UNDERSTANDING ===")
    print(f"Dimensioni iniziali del dataset: {df.shape}")
    print(f"Totale valori nulli rilevati: {df.isnull().sum().sum()}")
    print(f"Numero di righe duplicate: {df.duplicated().sum()}")

    # Controllo bilanciamento
    if target in df.columns:
        print("\nBilanciamento Target:")
        print(f" - Classe Depressi (1): {len(df[df[target] == 1])}")
        print(f" - Classe Non depressi (0): {len(df[df[target] == 0])}")


    # Controllo finale valori negativi
    num_cols = df.select_dtypes(include=['number']).columns
    neg_cols = [c for c in num_cols if (df[c] < 0).any()]

    if neg_cols:
        print(f"\nColonne con negativi: {neg_cols}")
    else:
        print("\nNessun valore negativo trovato.")

    # --- ANALISI FEATURE NUMERICHE ---
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in numerical_cols: numerical_cols.remove(target)

    numerical_corr = {}
    numerical_anomaly = {}

    for col in numerical_cols:
        col_data = df[col].dropna()
        numerical_corr[col] = col_data.corr(df[target])

        # Statistiche distribuzione
        skewness_val = skew(col_data)
        kurt_val = kurtosis(col_data)
        std_val = col_data.std()

        # Flag anomalie (tua logica originale)
        numerical_anomaly[col] = abs(skewness_val) > 1 or kurt_val > 5 or std_val < 0.01

    # --- ANALISI FEATURE CATEGORICHE ---
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_dependence = {}
    categorical_anomaly = {}

    for col in categorical_cols:
        col_data = df[col].dropna()
        # Calcolo dipendenza tramite crosstab
        ct = pd.crosstab(col_data, df[target], normalize='index')
        max_diff = abs(ct[1] - ct[0]).max()
        categorical_dependence[col] = max_diff

        # Flag distribuzione (tua logica originale)
        n_unique = col_data.nunique()
        value_counts = col_data.value_counts(normalize=True) * 100
        categorical_anomaly[col] = n_unique <= 1 or value_counts.max() > 90

    # --- STAMPE RISULTATI ANALISI ---
    sorted_corr = dict(sorted(numerical_corr.items(), key=lambda x: abs(x[1]), reverse=True))
    print("\nFeature numeriche più correlate alla target:")
    for col, corr in sorted_corr.items():
        print(f"{col}: {corr:.3f}")

    sorted_dep = dict(sorted(categorical_dependence.items(), key=lambda x: x[1], reverse=True))
    print("\nFeature categoriche con maggiore dipendenza dalla target:")
    for col, dep in sorted_dep.items():
        print(f"{col}: {dep:.3f}")

    # --- IDENTIFICAZIONE CANDIDATI ALLA RIMOZIONE ---
    numerical_candidates = [col for col in numerical_cols if numerical_anomaly[col] or abs(numerical_corr[col]) < 0.05]
    print("\nFeature numeriche anomale candidate per rimozione:", numerical_candidates)

    categorical_candidates = [col for col in categorical_cols if categorical_anomaly[col] or categorical_dependence[col] < 0.05]
    print("Feature categoriche anomale candidate per rimozione:", categorical_candidates)

    if 'Have you ever had suicidal thoughts ?' in df.columns:
        ct = pd.crosstab(
        df['Have you ever had suicidal thoughts ?'],
        df['Depression'],
        normalize='index'
        )
        print("\n", ct)

    # --- ESECUZIONE CLEANING ---
    print("\n=== ESECUZIONE DATA CLEANING ===")
    df_cleaned = getCleanedData(df)

    # Analisi Post-Cleaning
    print(f"Righe rimosse totali: {len(df) - len(df_cleaned)}")
    print(f"Dimensioni finali del dataset: {df_cleaned.shape}")

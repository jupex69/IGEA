import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# ==================
#   DATA CLEANING
# ==================


def getCleanedData(df_input):
    """
    Esegue SOLO l'azione di pulizia sui dati:
    - Rimozione colonne manuali
    - Rimozione Null e Duplicati
    - Rimozione Outlier (IQR 0.30 - 0.70)
    - Rimozione righe con CGPA < 4
    """
    # 1. Copia e parametri locali
    df = df_input.copy()
    target = 'Depression'

    # 2. Rimozione colonne non informative
    columns_to_drop = ['id', 'Have you ever had suicidal thoughts ?']
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

        # 5. Rimozione righe con CGPA < 4
        if 'CGPA' in df.columns:
            df = df[df['CGPA'] >= 4]

    return df
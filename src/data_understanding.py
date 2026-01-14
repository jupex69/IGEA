import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
 #  DATA UNDERSTANDING
 # ========================
if __name__ == "__main__":

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_PATH = BASE_DIR / "student_depression.csv"

    df = pd.read_csv(DATASET_PATH)

    target = 'Depression'

    # --- ANALISI ---
    print("\n=== DATA UNDERSTANDING ===")
    print(f"Dimensioni iniziali del dataset: {df.shape}")
    print(f"Totale valori nulli rilevati: {df.isnull().sum().sum()}")
    print(f"Numero di righe duplicate: {df.duplicated().sum()}")

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


    print("\n=== DISTRIBUZIONE DELLA TARGET ===")
    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True) * 100)

    print("\n=== ANALISI DISTRIBUZIONE FEATURE NUMERICHE ===")

    for col in numerical_cols:
        col_data = df[col].dropna()

        print(f"\n--- {col} ---")
        print(col_data.describe())
        print(f"Skewness: {skew(col_data):.3f}")
        print(f"Kurtosis: {kurtosis(col_data):.3f}")
        """
        # Istogramma
        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        sns.histplot(col_data, kde=True)
        plt.title(f"Distribuzione {col}")

        # Boxplot
        plt.subplot(1,3,2)
        sns.boxplot(x=col_data)
        plt.title(f"Boxplot {col}")

        # Distribuzione rispetto alla target
        plt.subplot(1,3,3)
        sns.boxplot(x=df[target], y=df[col])
        plt.title(f"{col} vs {target}")

        plt.tight_layout()
        plt.show()
        """

    print("\n=== ANALISI DISTRIBUZIONE FEATURE CATEGORICHE ===")

    for col in categorical_cols:
        col_data = df[col].dropna()

        print(f"\n--- {col} ---")
        print("Numero categorie:", col_data.nunique())
        print("Distribuzione percentuale:")
        print(col_data.value_counts(normalize=True) * 100)
        """
        plt.figure(figsize=(12,4))

        # Barplot generale
        plt.subplot(1,2,1)
        sns.countplot(x=col_data, order=col_data.value_counts().index)
        plt.title(f"Distribuzione {col}")
        plt.xticks(rotation=45)

        # Barplot rispetto alla target
        plt.subplot(1,2,2)
        ct = pd.crosstab(df[col], df[target], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
        """

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


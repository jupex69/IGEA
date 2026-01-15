import pandas as pd
import numpy as np
import sys
import warnings

# ------------------------------------------------------------------------------
# 0. CONFIGURAZIONE AMBIENTE
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# 1. CARICAMENTO DATI
# ==============================================================================
print("⏳ 1. Caricamento dati dalle pipeline...")

datasets = {}

# --- PIPELINE 1 (Assumiamo sia già ENCODED / SOLO NUMERICA) ---
try:
    import pipeline1
    if hasattr(pipeline1, 'X') and hasattr(pipeline1, 'y'):
        datasets['Pipeline 1'] = (pipeline1.X, pipeline1.y)
    elif hasattr(pipeline1, 'df'):
        target = 'Depression'
        datasets['Pipeline 1'] = (pipeline1.df.drop(columns=[target]), pipeline1.df[target])
except ImportError:
    print("⚠️  File 'pipeline1.py' non trovato.")

# --- PIPELINE 2 (Assumiamo sia GREZZA / MISTA TESTO-NUMERI) ---
try:
    import pipeline2
    if hasattr(pipeline2, 'X') and hasattr(pipeline2, 'y'):
        datasets['Pipeline 2'] = (pipeline2.X, pipeline2.y)
    elif hasattr(pipeline2, 'df'):
        target = 'Depression'
        datasets['Pipeline 2'] = (pipeline2.df.drop(columns=[target]), pipeline2.df[target])
except ImportError:
    print("⚠️  File 'pipeline2.py' non trovato.")

# --- SIMULAZIONE (SOLO SE NECESSARIO) ---
if not datasets:
    print("⚠️ Nessun dato trovato. Esco.")
    sys.exit()

# ==============================================================================
# 2. SETUP GENERALE
# ==============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
leaderboard_data = []

# ==============================================================================
# 3. CICLO DI ANALISI
# ==============================================================================

for pipe_name, (X, y) in datasets.items():

    print(f"\n" + "="*60)
    print(f"📂 ELABORAZIONE: {pipe_name}")
    print("="*60)

    # Colonne presenti in questo dataset specifico
    cols_in_dataset = X.columns.tolist()

    # Identificazione automatica tipi (serve solo per Pipeline 2 solitamente)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # LOGICA CONDIZIONALE PER IL PRE-PROCESSING
    # Se è Pipeline 1 -> Evitiamo l'encoding (è già fatto)
    # Se è Pipeline 2 -> Facciamo l'encoding
    is_already_encoded = (pipe_name == 'Pipeline 1')

    # --------------------------------------------------------------------------
    # A. DECISION TREE
    # --------------------------------------------------------------------------
    algo_name = "Decision Tree"
    print(f"\n🔹 Modello: {algo_name}")

    # Configurazione Scaler Custom (solo per le colonne richieste che esistono)
    target_scale_cols = ['Age', 'Study Satisfaction', 'Work/Study Hours', 'Stress_Amplified']
    cols_to_scale = [c for c in target_scale_cols if c in cols_in_dataset]

    transformers_tree = [
        # Scaliamo sempre e solo le colonne specifiche richieste
        ('scaler', StandardScaler(), cols_to_scale)
    ]

    # SE I DATI NON SONO ENCODED (Pipeline 2), AGGIUNGIAMO L'ENCODER
    if not is_already_encoded and len(cat_cols) > 0:
        transformers_tree.append(
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        )
        print("   -> Applico OrdinalEncoder (dati grezzi rilevati).")
    else:
        print("   -> Nessun Encoding applicato (dati assunti già numerici).")

    preprocess_tree = ColumnTransformer(
        transformers=transformers_tree,
        remainder='passthrough' # Il resto passa invariato
    )

    # Pipeline e GridSearch
    pipeline_tree = Pipeline(steps=[
        ('preprocess', preprocess_tree),
        ('model', DecisionTreeClassifier(class_weight='balanced', random_state=42))
    ])

    param_grid = {
        'model__max_depth': [None, 5, 10, 15, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__criterion': ['gini', 'entropy']
    }

    print("   ...Ricerca iperparametri (GridSearch)...")
    grid = GridSearchCV(pipeline_tree, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    best_tree = grid.best_estimator_

    # Validazione finale
    scores_tree = cross_validate(best_tree, X, y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], n_jobs=-1)

    # Stampa Risultati Tree
    print("   K-fold Cross Validation (5 fold) Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        val = scores_tree[f"test_{metric}"].mean()
        std = scores_tree[f"test_{metric}"].std()
        print(f"   {metric.capitalize():10s}: {val:.3f} ± {std:.3f}")

    leaderboard_data.append({
        'Source': pipe_name, 'Algorithm': algo_name,
        'Precision': scores_tree['test_precision'].mean(),
        'Accuracy': scores_tree['test_accuracy'].mean(),
        'Recall': scores_tree['test_recall'].mean(),
        'F1': scores_tree['test_f1'].mean(),
        'ROC AUC': scores_tree['test_roc_auc'].mean()
    })

    # --------------------------------------------------------------------------
    # B. LOGISTIC REGRESSION
    # --------------------------------------------------------------------------
    algo_name = "Logistic Regression"
    print(f"\n🔹 Modello: {algo_name}")

    transformers_log = []

    if is_already_encoded:
        # CASO PIPELINE 1: Dati già numerici.
        # Logistic Regression ha bisogno di TUTTO scalato, altrimenti non converge bene.
        # Non facciamo OneHot, ma scaliamo tutte le colonne.
        transformers_log.append(('num_all', StandardScaler(), cols_in_dataset))
        print("   -> Applico StandardScaler su tutto (dati già encoded).")
        remainder_log = 'passthrough' # (O drop, tanto abbiamo preso tutto)
    else:
        # CASO PIPELINE 2: Dati misti.
        # OneHot sulle categorie, Scaler sui numeri.
        transformers_log.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols))
        transformers_log.append(('num', StandardScaler(), num_cols))
        print("   -> Applico OneHotEncoder + StandardScaler (dati grezzi).")
        remainder_log = 'drop'

    preprocess_log = ColumnTransformer(transformers=transformers_log, remainder=remainder_log)

    pipeline_log = Pipeline(steps=[
        ('preprocess', preprocess_log),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    scores_log = cross_validate(pipeline_log, X, y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], n_jobs=-1)

    # Stampa Risultati LogReg
    print("   K-fold Cross Validation (5 fold) Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        val = scores_log[f"test_{metric}"].mean()
        std = scores_log[f"test_{metric}"].std()
        print(f"   {metric.capitalize():10s}: {val:.3f} ± {std:.3f}")

    leaderboard_data.append({
        'Source': pipe_name, 'Algorithm': algo_name,
        'Precision': scores_log['test_precision'].mean(),
        'Accuracy': scores_log['test_accuracy'].mean(),
        'Recall': scores_log['test_recall'].mean(),
        'F1': scores_log['test_f1'].mean(),
        'ROC AUC': scores_log['test_roc_auc'].mean()
    })

# ==============================================================================
# 4. LEADERBOARD FINALE
# ==============================================================================
if leaderboard_data:
    df_results = pd.DataFrame(leaderboard_data).sort_values(by='Precision', ascending=False)

    print("\n" + "="*80)
    print("🏆 CLASSIFICA FINALE (Ordinata per PRECISION)")
    print("="*80)
    print(df_results[['Source', 'Algorithm', 'Precision', 'Accuracy', 'Recall', 'F1', 'ROC AUC']].to_string(index=False, formatters={'Precision': '{:.4f}'.format, 'Accuracy': '{:.4f}'.format, 'Recall': '{:.4f}'.format, 'F1': '{:.4f}'.format, 'ROC AUC': '{:.4f}'.format}))

    winner = df_results.iloc[0]
    print("\n" + "-"*80)
    print(f"🥇 VINCITORE: {winner['Source']} + {winner['Algorithm']} (Precision: {winner['Precision']:.4f})")
    print("-" * 80)
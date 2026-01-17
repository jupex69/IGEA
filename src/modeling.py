import pandas as pd
import sys
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
import os
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==============================================================================
# 1. CARICAMENTO DATI
# ==============================================================================
print("⏳ Caricamento pipeline...")

datasets = {}

try:
    import pipeline1
    datasets['Pipeline 1'] = (pipeline1.X, pipeline1.y, pipeline1.preprocessor)
except ImportError:
    print("❌ pipeline1.py non trovato")
    sys.exit()

try:
    import pipeline2
    datasets['Pipeline 2'] = (pipeline2.X, pipeline2.y, pipeline2.preprocessor)
except ImportError:
    print("❌ pipeline2.py non trovato")
    sys.exit()

if not datasets:
    print("❌ Nessun dataset disponibile.")
    sys.exit()

# ==============================================================================
# 2. SETUP CV
# ==============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
leaderboard_data = []

# ==============================================================================
# 3. CICLO MODELLI
# ==============================================================================
for pipe_name, (X, y, preprocessor) in datasets.items():

    print("\n" + "=" * 60)
    print(f"📂 DATASET: {pipe_name}")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # A. DECISION TREE
    # --------------------------------------------------------------------------
    algo_name = "Decision Tree"
    print(f"\n🔹 Modello: {algo_name}")

    pipe_tree = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        ))
    ])

    print(" ...GridSearch in corso")
    param_grid = {
        'model__max_depth': [None, 8, 10, 15],
        'model__min_samples_split': [3, 5, 10],
        'model__min_samples_leaf': [5, 10, 14],
        'model__criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(
        pipe_tree,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X, y)
    best_tree = grid.best_estimator_
    print("\n✅ Migliore combinazione di iperparametri (Decision Tree)")
    for param, value in grid.best_params_.items():
        print(f"  {param}: {value}")

    scores_tree = cross_validate(
        best_tree,
        X, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1,
        return_estimator=True
    )

    # === STAMPE ORIGINALI ===
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f" {metric:10s}: {scores_tree[f'test_{metric}'].mean():.3f}")

    # === ANALISI K-FOLD ===
    print("\n📊 Analisi Fold-by-Fold (Decision Tree)")
    for i, acc in enumerate(scores_tree['test_accuracy']):
        print(f" Fold {i+1} accuracy: {acc:.3f}")
    print(f" Std accuracy: {scores_tree['test_accuracy'].std():.4f}")

    leaderboard_data.append({
        'Source': pipe_name,
        'Algorithm': algo_name,
        'Accuracy': scores_tree['test_accuracy'].mean(),
        'Precision': scores_tree['test_precision'].mean(),
        'Recall': scores_tree['test_recall'].mean(),
        'F1': scores_tree['test_f1'].mean(),
        'ROC AUC': scores_tree['test_roc_auc'].mean()
    })

    # === ANALISI REGOLE DECISIONALI ===
    print("\n📜 Regole decisionali (Decision Tree)")
    tree_model = best_tree.named_steps['model']
    feature_names = best_tree.named_steps['preprocess'].get_feature_names_out()
    #rules = export_text(tree_model, feature_names=feature_names)
    #print(rules)

    #print("\n🌳 Visualizzazione Decision Tree (profondità limitata)")
    #
    #plt.figure(figsize=(20, 10))
    #plot_tree(
    #    tree_model,
    #    feature_names=feature_names,
    #    class_names=['No Depression', 'Depression'],
    #    filled=True,
    #    rounded=True,
    #    fontsize=9,
    #    max_depth=3     # ← SOLO VISUALIZZAZIONE
    #)
    #plt.title(f"Decision Tree (depth ≤ 3) – {pipe_name}")
    #plt.show()


# === FEATURE IMPORTANCE ===
    print("\n⭐ Feature Importance (Decision Tree)")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': tree_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(importance_df)

    # --------------------------------------------------------------------------
    # B. LOGISTIC REGRESSION
    # --------------------------------------------------------------------------
    algo_name = "Logistic Regression"
    print(f"\n🔹 Modello: {algo_name}")

    pipe_log = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=15,
            random_state=42
        ))
    ])

    scores_log = cross_validate(
        pipe_log,
        X, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1,
        return_estimator=True
    )

    """
    print("\n📊 Confusion Matrix aggregata (Logistic Regression)")

    cm_total = np.zeros((2, 2), dtype=int)

    for fold_idx, estimator in enumerate(scores_log['estimator']):
        train_idx, test_idx = list(cv.split(X, y))[fold_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        y_pred = estimator.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        cm_total += cm

    print(cm_total)

    # === LABEL PIPELINE ===
    pipeline_label = (
        f"Pipeline: {pipe_name}\n"
        f"Model: {algo_name}"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_total,
        display_labels=['No Depression', 'Depression']
    )

    disp.plot(cmap='Blues')
    plt.title("Aggregated Confusion Matrix", fontsize=13)
    plt.suptitle(pipeline_label, fontsize=10)
    plt.tight_layout()


    # Percorso base
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "docs", "images")

    # Crea cartella se manca
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Nome file "sicuro" (senza spazi, senza caratteri strani)
    safe_pipe = pipe_name.replace(" ", "_")
    safe_algo = algo_name.replace(" ", "_")
    filename = os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_pipe}_{safe_algo}.png")

    # Salva il grafico
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    """

    # === STAMPE RISULTATI ===
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f" {metric:10s}: {scores_log[f'test_{metric}'].mean():.3f}")

    leaderboard_data.append({
        'Source': pipe_name,
        'Algorithm': algo_name,
        'Accuracy': scores_log['test_accuracy'].mean(),
        'Precision': scores_log['test_precision'].mean(),
        'Recall': scores_log['test_recall'].mean(),
        'F1': scores_log['test_f1'].mean(),
        'ROC AUC': scores_log['test_roc_auc'].mean()
    })

    # === ANALISI COEFFICIENTI BETA ===
    print("\n📈 Coefficienti Beta (Logistic Regression)")
    pipe_log.fit(X, y)

    model = pipe_log.named_steps['model']
    preprocess = pipe_log.named_steps['preprocess']
    feature_names = preprocess.get_feature_names_out()

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Beta': model.coef_[0],
        'Odds_Ratio': np.exp(model.coef_[0])
    }).sort_values(by='Beta', key=abs, ascending=False)

    print(coef_df)

# ==============================================================================
# 4. LEADERBOARD FINALE
# ==============================================================================
df_results = pd.DataFrame(leaderboard_data).sort_values(by='Precision', ascending=False)

print("\n" + "=" * 80)
print("🏆 CLASSIFICA FINALE")
print("=" * 80)
print(df_results.to_string(index=False, float_format="%.4f"))

# ==============================================================================
# 5. VISUALIZZAZIONE GRAFICA DEI RISULTATI
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

print("\n📊 Generazione grafico riassuntivo...")

# 1. Preparazione Dati per il Plot (Trasformazione in formato "Long")
# Creiamo una colonna unica per identificare la combinazione Pipeline + Modello
df_results['Model_ID'] = df_results['Source'] + "\n" + df_results['Algorithm']

# "Sciogliamo" il dataframe per avere le metriche sulle righe (formato ideale per seaborn)
df_long = df_results.melt(
    id_vars=['Model_ID'],
    value_vars=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
    var_name='Metrica',
    value_name='Score'
)

# 2. Configurazione Plot
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

# Creazione Barplot
ax = sns.barplot(
    data=df_long,
    x='Metrica',
    y='Score',
    hue='Model_ID',
    palette='viridis' # Usa 'rocket', 'mako' o 'viridis'
)

# 3. Etichette e Titoli
plt.title('Confronto Performance: Pipeline 1 vs Pipeline 2', fontsize=16, pad=20)
plt.ylabel('Punteggio (Score)', fontsize=12)
plt.xlabel('Metrica di Valutazione', fontsize=12)
plt.legend(title='Configurazione Modello', bbox_to_anchor=(1.02, 1), loc='upper left')

# 4. Zoom sull'asse Y per evidenziare le differenze
# Dato che i valori sono tutti vicini (0.77 - 0.85), partiamo da 0.70 per vedere meglio il distacco
plt.ylim(0.70, 0.90)

# 5. Aggiunta dei valori sopra le barre
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, rotation=90)

plt.tight_layout()
plt.show()


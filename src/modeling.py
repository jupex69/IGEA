import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
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
    print("❌ pipeline2.py non trovato")
    sys.exit()

try:
    import pipeline2
    datasets['Pipeline 2'] = (pipeline2.X, pipeline2.y, pipeline2.preprocessor)
except ImportError:
    print("❌ pipeline2.py non trovato")
    sys.exit()

if not datasets: print("❌ Nessun dataset disponibile.")
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
    algo_name="Decision Tree"
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
        'model__max_depth': [None, 5, 10, 15],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
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

    scores_tree = cross_validate(
        best_tree,
        X, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1
    )

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f" {metric:10s}:{scores_tree[f'test_{metric}'].mean():.3f}")

    leaderboard_data.append({
        'Source': pipe_name,
        'Algorithm': algo_name,
        'Accuracy': scores_tree['test_accuracy'].mean(),
        'Precision': scores_tree['test_precision'].mean(),
        'Recall': scores_tree['test_recall'].mean(),
        'F1': scores_tree['test_f1'].mean(),
        'ROC AUC': scores_tree['test_roc_auc'].mean()
    })

    # --------------------------------------------------------------------------
    # B. LOGISTIC REGRESSION
    # --------------------------------------------------------------------------
    algo_name = "Logistic Regression"
    print(f"\n🔹 Modello: {algo_name}")

    pipe_log = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])

    scores_log = cross_validate(
        pipe_log,
        X, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1
    )

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

# # ==============================================================================
# 4. LEADERBOARD FINALE
# ==============================================================================
df_results = pd.DataFrame(leaderboard_data).sort_values(by='Precision', ascending=False)
print("\n" + "=" * 80)
print("🏆 CLASSIFICA FINALE")
print("=" * 80)
print(df_results.to_string(index=False, float_format="%.4f"))

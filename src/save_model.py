import sys
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Aggiungiamo la directory corrente al path per importare i tuoi moduli
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importiamo la tua pipeline (assicurati che pipeline2.py sia nella stessa cartella o accessibile)
import pipeline2

print("⏳ Recupero dati dalla Pipeline 2...")
X, y, preprocessor = pipeline2.X, pipeline2.y, pipeline2.preprocessor

# Creiamo il modello finale: Preprocessor + Classificatore
# Usiamo LogisticRegression perché è solida e veloce per questo tipo di dati
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

print("⏳ Addestramento modello completo...")
final_model.fit(X, y)

# Salviamo il file nella stessa cartella src
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modello_depression.pkl')
joblib.dump(final_model, output_path)
print(f"✅ Modello salvato con successo in: {output_path}")
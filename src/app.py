from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
import numpy as np

# =========================
# CONFIGURAZIONE PERCORSI
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '../sito/templates')
STATIC_DIR = os.path.join(BASE_DIR, '../sito/static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# =========================
# CARICAMENTO MODELLO
# =========================
model_path = os.path.join(BASE_DIR, 'modello_depression.pkl')

try:
    model = joblib.load(model_path)
    print(f"✅ Modello caricato correttamente. (Numpy: {np.__version__})")
except Exception as e:
    print(f"❌ Errore caricamento modello: {e}")

# =========================
# FUNZIONI DI PULIZIA
# =========================
def clean_sleep_duration(val):
    """Rimuove apici e spazi extra"""
    return str(val).replace("'", "").strip()

def map_degree(val):
    """Normalizza il titolo di studio"""
    val = str(val).lower()
    if any(x in val for x in ['b.tech', 'b.e', 'bachelor', 'undergraduate']):
        return 'Bachelor'
    if any(x in val for x in ['m.tech', 'm.e', 'master', 'postgraduate']):
        return 'Master'
    if 'diploma' in val:
        return 'Diploma'
    if 'phd' in val:
        return 'PhD'
    return 'Bachelor' # Default

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('questionario.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # 1. Recupero dati grezzi dal form
    input_data = {
        'Gender': str(data.get('gender', 'Male')),
        'Age': float(data.get('age', 20)),
        'Academic Pressure': float(data.get('academic_pressure', 3)),
        'Study Satisfaction': float(data.get('study_satisfaction', 3)),
        'Financial Stress': float(data.get('financial_stress', 3)),
        'Degree': str(data.get('degree', 'Diploma')),
        'CGPA': float(data.get('cgpa', 8.0)),
        'Work/Study Hours': float(data.get('study_hours', 4)),
        'Sleep Duration': str(data.get('sleep_duration', '7-8 hours')),
        'Dietary Habits': str(data.get('dietary_habits', 'Moderate')),
        'Family History of Mental Illness': str(data.get('family_history_mental_illness', 'No'))
    }

    input_df = pd.DataFrame([input_data])

    # 2. PULIZIA DATI
    input_df['Sleep Duration'] = input_df['Sleep Duration'].apply(clean_sleep_duration)
    input_df['Degree'] = input_df['Degree'].apply(map_degree)

    # 3. FEATURE ENGINEERING (Creazione colonne mancanti)
    input_df['CGPA_30'] = (input_df['CGPA'] * 2.4) + 6
    input_df['Degree_level'] = input_df['Degree']

    # 4. FORCE TYPE CASTING (La soluzione all'errore isnan)
    # Convertiamo esplicitamente le colonne di testo in 'object' per calmare Numpy
    string_cols = ['Gender', 'Degree', 'Degree_level', 'Sleep Duration', 'Dietary Habits', 'Family History of Mental Illness']
    for col in string_cols:
        input_df[col] = input_df[col].astype(object)

    # Assicuriamo che i numeri siano float
    numeric_cols = ['Age', 'Academic Pressure', 'Study Satisfaction', 'Financial Stress', 'CGPA', 'Work/Study Hours', 'CGPA_30']
    for col in numeric_cols:
        input_df[col] = input_df[col].astype(float)

    # 5. PREDIZIONE
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            msg = "Rischio Depressione Rilevato"
            color = "#dc3545"
        else:
            msg = "Nessun Rischio Rilevato"
            color = "#28a745"

        return render_template(
            'risultato.html',
            prediction=msg,
            prob=round(prob * 100, 1),
            color=color
        )

    except Exception as e:
        print(f"\n❌ ERRORE PREDIZIONE: {e}")
        import traceback
        traceback.print_exc() # Stampa l'errore completo nel terminale per debug
        return f"Errore del server: {e}"

if __name__ == '__main__':
    app.run(debug=True)
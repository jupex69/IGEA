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
model_path = os.path.join(BASE_DIR, 'modello_depressione_finale.pkl')
try:
    model = joblib.load(model_path)
    print("✅ Modello caricato correttamente.")
except Exception as e:
    print(f"❌ Errore caricamento modello: {e}")

# =========================
# FUNZIONI DI TRASFORMAZIONE (Identiche a Pipeline 2)
# =========================

def map_degree_pipeline2(deg):
    """
    Replica la logica di pipeline2.py per Degree_level
    """
    deg = str(deg).strip()
    if deg == "'Class 12'" or 'Diploma' in deg:
        return 'Diploma'
    elif deg.startswith('B') or 'Bachelor' in deg:
        return 'Titolo_primo_livello'
    elif deg.startswith('M') or 'Master' in deg or deg in ['PhD', 'MD']:
        return 'Titolo_secondo_livello'
    else:
        return 'Titolo_secondo_livello'  # fallback sicuro

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/questionario')
def questionario():
    return render_template('questionario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # -----------------------------
        # 1. RAW INPUT
        # -----------------------------
        raw_input = {
            'Gender': str(data.get('gender', 'Male')),
            'Age': float(data.get('age', 20)),
            'Academic Pressure': float(data.get('academic_pressure', 3)),
            'Study Satisfaction': float(data.get('study_satisfaction', 3)),
            'Financial Stress': str(float(data.get('financial_stress', 3))),
            'Degree_level': str(data.get('degree', 'Diploma')),
            'CGPA_30': float(data.get('cgpa', 25)),
            'Work/Study Hours': float(data.get('study_hours', 4)),
            'Sleep Duration': str(data.get('sleep_duration', "'7-8 hours'")),
            'Dietary Habits': str(data.get('dietary_habits', 'Moderate')),
            'Family History of Mental Illness': str(data.get('family_history_mental_illness', 'No'))
        }

        input_df = pd.DataFrame([raw_input])

        categorical_cols = ['Gender', 'Sleep Duration', 'Dietary Habits',
                            'Family History of Mental Illness', 'Degree_level']
        numeric_cols = ['Age', 'Academic Pressure', 'Study Satisfaction',
                        'Financial Stress', 'Work/Study Hours', 'CGPA_30']

        input_df = input_df[categorical_cols + numeric_cols]

        # -----------------------------
        # 4. PREDICTION
        # -----------------------------
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            msg = "Rischio Depressione Rilevato"
            color = "#dc3545"
            tips = "Ti consigliamo di parlarne con uno specialista."
        else:
            msg = "Nessun Rischio Rilevato"
            color = "#28a745"
            tips = "Continua così!"

        return render_template(
            'risultato.html',
            prediction=msg,
            prob=round(prob * 100, 1),
            color=color,
            tips=tips
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h3>Errore del sistema:</h3><p>{e}</p>"

# =========================
if __name__ == '__main__':
    app.run(debug=True)

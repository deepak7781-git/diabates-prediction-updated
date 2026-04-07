"""
app.py  —  Flask backend for DiabetesGuard
==========================================
Endpoints:
  GET  /          → serves index.html
  POST /predict   → accepts JSON, returns risk probability + warning signs
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── App setup ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load model ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")

bundle   = joblib.load(MODEL_PATH)
model    = bundle["model"]
FEATURES = bundle["features"]
ZERO_COLS = bundle["zero_cols"]

# ── Clinical thresholds for warning signs ─────────────────────────
THRESHOLDS = {
    "Glucose": {
        "normal":  (70,  99),
        "warning": (100, 125),
        "danger":  (126, 9999),
        "unit": "mg/dL",
        "hint_warning": "Pre-diabetic glucose level (100–125 mg/dL)",
        "hint_danger":  "Diabetic glucose level (≥126 mg/dL)",
    },
    "BloodPressure": {
        "normal":  (60,  79),
        "warning": (80,  89),
        "danger":  (90,  9999),
        "unit": "mmHg",
        "hint_warning": "Elevated blood pressure (80–89 mmHg)",
        "hint_danger":  "High blood pressure ≥90 mmHg — hypertension",
    },
    "SkinThickness": {
        "normal":  (0,  27),
        "warning": (28, 35),
        "danger":  (36, 999),
        "unit": "mm",
        "hint_warning": "Slightly elevated skinfold — possible insulin resistance",
        "hint_danger":  "High skinfold thickness — strong indicator of insulin resistance",
    },
    "Insulin": {
        "normal":  (2,  25),
        "warning": (26, 50),
        "danger":  (51, 99999),
        "unit": "μU/mL",
        "hint_warning": "Elevated fasting insulin — possible insulin resistance",
        "hint_danger":  "Very high insulin level — significant insulin resistance",
    },
    "BMI": {
        "normal":  (18.5, 24.9),
        "warning": (25.0, 29.9),
        "danger":  (30.0, 999),
        "unit": "kg/m²",
        "hint_warning": "Overweight (BMI 25–29.9) — increased diabetes risk",
        "hint_danger":  "Obese (BMI ≥30) — high diabetes risk",
    },
    "DiabetesPedigreeFunction": {
        "normal":  (0.0, 0.40),
        "warning": (0.41, 0.60),
        "danger":  (0.61, 9.99),
        "unit": "score",
        "hint_warning": "Moderate genetic risk from family history",
        "hint_danger":  "High genetic predisposition to diabetes",
    },
    "Age": {
        "normal":  (0,  34),
        "warning": (35, 44),
        "danger":  (45, 999),
        "unit": "years",
        "hint_warning": "Age 35–44: moderate diabetes risk window",
        "hint_danger":  "Age ≥45: significantly higher diabetes risk",
    },
}

def classify_field(field, value):
    """Returns status string and hint."""
    if field not in THRESHOLDS or value is None:
        return "unknown", "Value not provided — estimated by model"
    t = THRESHOLDS[field]
    lo_d, hi_d = t["danger"]
    lo_w, hi_w = t["warning"]
    if lo_d <= value <= hi_d:
        return "danger",  t["hint_danger"]
    elif lo_w <= value <= hi_w:
        return "warning", t["hint_warning"]
    else:
        return "normal",  f"Within normal range ({t['unit']})"


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
      "Glucose": 148,
      "BloodPressure": 72,
      "SkinThickness": 35,
      "Insulin": 0,
      "BMI": 33.6,
      "DiabetesPedigreeFunction": 0.627,
      "Age": 50
    }

    Returns:
    {
      "probability": 73.4,
      "risk_level": "HIGH",
      "risk_color": "#e74c3c",
      "warnings": [...],
      "field_analysis": [...]
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Validate all required fields exist
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Build input — replace 0 in medical cols with NaN (imputed by model)
        patient = {}
        for f in FEATURES:
            v = data.get(f)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = np.nan
            if f in ZERO_COLS and v == 0.0:
                v = np.nan
            patient[f] = v

        df_in = pd.DataFrame([patient])

        # Predict
        prob = float(model.predict_proba(df_in)[0][1])
        pct  = round(prob * 100, 1)

        # Risk bucket — using lower threshold (0.35) for medical screening
        if pct < 30:
            risk_level = "LOW"
            risk_color = "#27ae60"
            risk_emoji = "✅"
        elif pct < 50:
            risk_level = "MODERATE"
            risk_color = "#f39c12"
            risk_emoji = "⚠️"
        elif pct < 72:
            risk_level = "HIGH"
            risk_color = "#e67e22"
            risk_emoji = "🟠"
        else:
            risk_level = "VERY HIGH"
            risk_color = "#e74c3c"
            risk_emoji = "🚨"

        # Per-field analysis
        field_analysis = []
        warnings_list  = []
        for f in FEATURES:
            raw_val = data.get(f)
            try:
                raw_val = float(raw_val)
            except (TypeError, ValueError):
                raw_val = None

            status, hint = classify_field(f, raw_val if raw_val and raw_val > 0 else None)
            unit = THRESHOLDS.get(f, {}).get("unit", "")
            display_val = f"{raw_val} {unit}" if raw_val and raw_val > 0 else "Not provided"

            field_analysis.append({
                "field":   f,
                "value":   display_val,
                "status":  status,
                "hint":    hint,
            })

            if status in ("warning", "danger"):
                warnings_list.append({
                    "field":    f,
                    "severity": status,
                    "hint":     hint,
                    "value":    display_val,
                })

        return jsonify({
            "probability":    pct,
            "risk_level":     risk_level,
            "risk_color":     risk_color,
            "risk_emoji":     risk_emoji,
            "warnings":       warnings_list,
            "field_analysis": field_analysis,
            "model_used":     bundle.get("model_name", "ML Model"),
            "test_auc":       bundle.get("test_auc", "N/A"),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": bundle.get("model_name"),
        "features": FEATURES,
        "test_auc": bundle.get("test_auc"),
    })


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)

# DiabetesGuard — Flask + ML App

## Folder Structure

```
diabetes_app/
│
├── app.py                        ← Flask server (run this)
├── requirements.txt              ← Python dependencies
├── diabetes_model.pkl            ← Generated after training (do not commit)
│
├── model_training/
│   ├── train_model.py            ← Run once to train & save the model
│   └── diabetes.csv              ← Dataset
│
└── templates/
    └── index.html                ← Frontend (auto-served by Flask)
```

---

## Step-by-Step Setup

### 1. Create & activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (run ONCE)

```bash
python model_training/train_model.py
```

This will:
- Load `diabetes.csv`, train the best ML model
- Apply probability calibration
- Save `diabetes_model.pkl` in the project root
- Print test AUC and classification report

### 4. Start the Flask server

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

### 5. Open the app

Open your browser and go to:
```
http://localhost:5000
```

---

## API Endpoints

| Method | URL        | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Serves the frontend                |
| POST   | `/predict` | Runs ML prediction, returns JSON   |
| GET    | `/health`  | Model info and status check        |

### Example `/predict` request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

### Example response

```json
{
  "probability": 73.4,
  "risk_level": "HIGH",
  "risk_color": "#e67e22",
  "risk_emoji": "🟠",
  "model_used": "Logistic Regression",
  "test_auc": 0.8414,
  "warnings": [
    {
      "field": "Glucose",
      "severity": "danger",
      "hint": "Diabetic glucose level (≥126 mg/dL)",
      "value": "148 mg/dL"
    }
  ],
  "field_analysis": [ ... ]
}
```

---

## Notes

- `Insulin` and `SkinThickness` can be sent as `0` if unknown — the model will impute them automatically using KNN.
- The model uses **probability calibration** (isotonic regression) for accurate percentage outputs.
- This is a screening tool only. Not a clinical diagnosis.

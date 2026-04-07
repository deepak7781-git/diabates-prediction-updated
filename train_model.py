"""
train_model.py
==============
Run this ONCE to train the model and save diabetes_model.pkl
Usage:  python model_training/train_model.py
Output: diabetes_model.pkl  (saved in project root)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ── Paths ───────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")
MODEL_OUT = os.path.join(ROOT_DIR, "diabetes_model.pkl")

# ── Load data ───────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.drop(columns=["Pregnancies"], inplace=True)

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

FEATURES = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Preprocessing ────────────────────────────────────────────────
preprocessor = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler",  RobustScaler()),
])

# ── Model comparison ─────────────────────────────────────────────
print("=" * 55)
print("CROSS-VALIDATION  (5-fold, ROC-AUC)")
print("=" * 55)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05,
                                         scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                         eval_metric="logloss", random_state=42, verbosity=0),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = {}
for name, mdl in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", mdl)])
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    scores[name] = s.mean()
    print(f"  {name:<25} AUC: {s.mean():.4f} ± {s.std():.4f}")

best_name = max(scores, key=scores.get)
print(f"\n  Best model: {best_name}  (AUC {scores[best_name]:.4f})")

# ── Build best pipeline + calibrate probabilities ─────────────────
best_clf = models[best_name]
final_pipeline = Pipeline([
    ("prep", preprocessor),
    # CalibratedClassifierCV fixes the miscalibrated probabilities flaw
    ("clf", CalibratedClassifierCV(best_clf, method="isotonic", cv=5))
])

final_pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────
y_pred  = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "=" * 55)
print("TEST SET RESULTS")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── Save model + metadata ─────────────────────────────────────────
bundle = {
    "model":    final_pipeline,
    "features": FEATURES,
    "zero_cols": ZERO_COLS,
    "model_name": best_name,
    "test_auc": round(roc_auc_score(y_test, y_proba), 4),
}
joblib.dump(bundle, MODEL_OUT)
print(f"\n  Model saved → {MODEL_OUT}")

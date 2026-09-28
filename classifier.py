import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss

LABELS = {0: "하락", 1: "횡보", 2: "상승"}

def fit_model(X, y, calibrate=True):
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ))
    ])
    if calibrate and len(X) >= 250 and y.nunique() == 3:
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        model = base
    model.fit(X, y)
    return model

def predict(model, X):
    p = model.predict_proba(X)[0]
    classes = model.classes_
    out = {"하락": 0.0, "횡보": 0.0, "상승": 0.0}
    for c, prob in zip(classes, p):
        out[LABELS[int(c)]] = float(prob)
    return out

def diagnostics(model, X, y):
    pred = model.predict(X)
    return {
        "Accuracy": round(float(accuracy_score(y, pred)), 3),
        "Balanced Accuracy": round(float(balanced_accuracy_score(y, pred)), 3),
        "표본": int(len(y)),
    }

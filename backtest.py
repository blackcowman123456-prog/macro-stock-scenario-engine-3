import numpy as np
import pandas as pd
from models.classifier import fit_model, predict

def walk_forward(X, y, min_train=180, step=1, calibrate=False):
    rows = []
    for i in range(min_train, len(X), step):
        Xtr, ytr = X.iloc[:i], y.iloc[:i]
        if ytr.nunique() < 3:
            continue
        model = fit_model(Xtr, ytr, calibrate=calibrate)
        p = predict(model, X.iloc[[i]])
        actual = int(y.iloc[i])
        pred_label = {"하락":0, "횡보":1, "상승":2}[max(p, key=p.get)]
        rows.append({
            "date": X.index[i],
            "actual": actual,
            "predicted": pred_label,
            "hit": int(pred_label == actual),
            "상승확률": p["상승"],
            "횡보확률": p["횡보"],
            "하락확률": p["하락"],
        })
    return pd.DataFrame(rows)

def walk_forward_report(X, y, min_train=180, calibrate=False):
    df = walk_forward(X, y, min_train=min_train, calibrate=calibrate)
    if df.empty:
        return df
    return df.tail(24).assign(
        정확도=lambda d: d["hit"].expanding().mean().round(3)
    )

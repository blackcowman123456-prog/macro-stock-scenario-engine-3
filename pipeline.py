import numpy as np
import pandas as pd

from providers.market import load_market, MARKET_CONFIG
from providers.fred import load_us_macro
from providers.ecos import get_metric
from core.features import build_features, make_label
from models.classifier import fit_model, predict, diagnostics
from core.backtest import walk_forward_report

def _load_korea_macro(key, start):
    # ECOS table discovery is deliberately used rather than hard-coding fragile table codes.
    # If an item is not found, the pipeline keeps the remaining available features.
    specs = [
        ("기준금리", ["기준금리"]),
        ("소비자물가", ["소비자물가"]),
        ("생산자물가", ["생산자물가"]),
        ("M2", ["M2"]),
    ]
    out = []
    for name, kw in specs:
        try:
            s = get_metric(key, kw, cycle="MM", start=pd.Timestamp(start).strftime("%Y%m"))
            s.name = name
            out.append(s)
        except Exception:
            pass
    if not out:
        raise RuntimeError("ECOS 데이터를 가져오지 못했습니다. ECOS API Key와 API 접근권한을 확인하세요.")
    return pd.concat(out, axis=1).resample("ME").last()

def _load_macro(market, key, start):
    if market == "미국":
        return load_us_macro(key, start)
    return _load_korea_macro(key, start)

def _stock_predictions(market, stock_prices, features, horizon, min_train, calibrate):
    rows = []
    for name, ticker in MARKET_CONFIG[market]["stocks"].items():
        if ticker not in stock_prices.columns:
            continue
        price = stock_prices[ticker].dropna()
        y, fwd = make_label(price, horizon)
        common = features.index.intersection(y.index)
        X = features.reindex(common)
        yy = y.reindex(common)
        valid = yy.notna()
        X, yy = X.loc[valid], yy.loc[valid].astype(int)
        if len(X) < min_train or yy.nunique() < 3:
            continue
        model = fit_model(X.iloc[:-1], yy.iloc[:-1], calibrate=calibrate)
        p = predict(model, X.iloc[[-1]])
        recent = fwd.reindex(common).dropna().tail(60)
        rows.append({
            "종목": name,
            "티커": ticker,
            "상승확률": p["상승"],
            "횡보확률": p["횡보"],
            "하락확률": p["하락"],
            "예상수익률 중앙값": float(recent.median()) if not recent.empty else np.nan,
        })
    return pd.DataFrame(rows)

def run_pipeline(market, target, horizon, train_years, settings, min_train=180, calibrate=True):
    start = pd.Timestamp.today() - pd.DateOffset(years=train_years + 2)
    macro = _load_macro(market, settings.fred_key if market == "미국" else settings.ecos_key, start)
    market_prices = load_market(market, start)

    features = build_features(market_prices, macro, target)
    price = market_prices[target].dropna()
    y, fwd = make_label(price, horizon)

    idx = features.index.intersection(y.index)
    X = features.reindex(idx)
    yy = y.reindex(idx)
    valid = yy.notna()
    X, yy = X.loc[valid], yy.loc[valid].astype(int)

    if len(X) < min_train or yy.nunique() < 3:
        raise RuntimeError(
            f"학습 데이터가 부족합니다. 현재 {len(X)}개 표본이며 최소 {min_train}개가 필요합니다."
        )

    # Last row is current prediction point. Training ends strictly before it.
    X_train, y_train = X.iloc[:-1], yy.iloc[:-1]
    model = fit_model(X_train, y_train, calibrate=calibrate)
    probs = predict(model, X.iloc[[-1]])
    diag = diagnostics(model, X_train, y_train)

    bt = walk_forward_report(
        X_train, y_train, min_train=min_train, calibrate=False
    )

    stock_pred = _stock_predictions(
        market, market_prices, features, horizon, min_train, calibrate
    )

    return {
        "asof": X.index[-1],
        "n_train": len(X_train),
        "target_label": MARKET_CONFIG[market]["labels"].get(target, target),
        "probabilities": probs,
        "diagnostics": diag,
        "current_features": X.iloc[-1].dropna(),
        "stock_predictions": stock_pred,
        "backtest": bt,
        "live": True,
    }

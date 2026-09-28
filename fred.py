import requests
import pandas as pd

BASE = "https://api.stlouisfed.org/fred/series/observations"

def get_series(api_key, series_id, start="2000-01-01", end=None, frequency=None):
    if not api_key:
        raise ValueError("FRED_API_KEY가 필요합니다.")
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end or pd.Timestamp.today().strftime("%Y-%m-%d"),
        "sort_order": "asc",
    }
    if frequency:
        params["frequency"] = frequency
        params["aggregation_method"] = "eop"
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("observations", [])
    if not data:
        raise ValueError(f"FRED series {series_id}에 데이터가 없습니다.")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].rename(series_id).dropna()

def load_us_macro(api_key, start):
    ids = {
        "fedfunds": "FEDFUNDS",
        "cpi": "CPIAUCSL",
        "core_cpi": "CPILFESL",
        "pce": "PCEPI",
        "unrate": "UNRATE",
        "payems": "PAYEMS",
        "indpro": "INDPRO",
        "m2": "M2SL",
        "hy_spread": "BAMLH0A0HYM2",
        "10y": "DGS10",
        "2y": "DGS2",
        "vix": "VIXCLS",
        "real_gdp": "GDPC1",
    }
    out = []
    for name, sid in ids.items():
        try:
            s = get_series(api_key, sid, start=start)
            s.name = name
            out.append(s)
        except Exception:
            pass
    if not out:
        raise RuntimeError("FRED에서 사용할 수 있는 시계열을 하나도 가져오지 못했습니다.")
    df = pd.concat(out, axis=1).resample("ME").last()
    return df

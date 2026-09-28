import yfinance as yf
import pandas as pd
import numpy as np

MARKET_CONFIG = {
    "한국": {
        "targets": ["^KS11", "^KQ11"],
        "labels": {"^KS11": "KOSPI", "^KQ11": "KOSDAQ"},
        "stocks": {
            "삼성전자": "005930.KS",
            "SK하이닉스": "000660.KS",
            "현대차": "005380.KS",
            "NAVER": "035420.KS",
            "KB금융": "105560.KS",
        },
    },
    "미국": {
        "targets": ["^GSPC", "^NDX"],
        "labels": {"^GSPC": "S&P 500", "^NDX": "Nasdaq 100"},
        "stocks": {
            "Microsoft": "MSFT",
            "NVIDIA": "NVDA",
            "JPMorgan": "JPM",
            "Amazon": "AMZN",
            "Alphabet": "GOOGL",
        },
    },
}

def get_price(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"{ticker} 시장 데이터 없음")
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna().rename(ticker)

def load_market(market, start):
    cfg = MARKET_CONFIG[market]
    series = []
    for ticker in cfg["targets"]:
        series.append(get_price(ticker, start).resample("ME").last())
    for ticker in cfg["stocks"].values():
        try:
            series.append(get_price(ticker, start).resample("ME").last())
        except Exception:
            pass
    return pd.concat(series, axis=1)

import streamlit as st
import pandas as pd
import numpy as np

from core.pipeline import run_pipeline
from core.settings import load_settings
from core.backtest import walk_forward_report
from providers.market import MARKET_CONFIG

st.set_page_config(page_title="Macro → Stock Forecast Engine", page_icon="📈", layout="wide")

st.title("📈 Macro → Stock Forecast Engine")
st.caption("한국/미국 거시환경 → 시장 국면 → 3·6·12개월 방향성 → 종목별 확률")

with st.sidebar:
    st.header("설정")
    market = st.radio("시장", ["한국", "미국"], horizontal=True)
    horizon = st.selectbox("예측 기간", [3, 6, 12], index=0, format_func=lambda x: f"{x}개월")
    train_years = st.slider("학습 기간(년)", 5, 20, 10)
    target = st.selectbox(
        "시장 타깃",
        MARKET_CONFIG[market]["targets"],
        format_func=lambda x: MARKET_CONFIG[market]["labels"].get(x, x),
    )
    run = st.button("🚀 현재 데이터로 예측", type="primary", use_container_width=True)

    st.divider()
    st.subheader("API 키")
    st.caption("키는 GitHub 코드에 저장하지 마세요. Streamlit Secrets를 우선 사용합니다.")
    fred_key = st.text_input("FRED API Key (미국)", type="password")
    ecos_key = st.text_input("ECOS API Key (한국)", type="password")

    with st.expander("고급"):
        min_train = st.number_input("최소 학습 표본", 120, 1000, 180, 10)
        probability_bins = st.checkbox("확률 Calibration 적용", value=True)

settings = load_settings(fred_key=fred_key, ecos_key=ecos_key)

st.info(
    "이 앱은 통계적 예측을 위한 연구용 모델입니다. 예측 확률은 보장된 결과가 아니며 "
    "투자 의사결정의 유일한 근거로 사용하지 마세요."
)

if not run:
    st.markdown(
        """
### 사용 순서
1. 시장을 한국/미국 중 선택합니다.
2. FRED/ECOS API 키를 입력하거나 Streamlit Secrets에 등록합니다.
3. **현재 데이터로 예측**을 누릅니다.
4. 모델이 과거 데이터를 자동 수집하고, 현재 시점까지의 데이터로 학습한 뒤 3/6/12개월 방향성을 계산합니다.
5. 아래에서 예측 확률과 최근 백테스트 성능을 확인합니다.

**중요:** 경제지표는 발표·수정 시점이 있기 때문에, 완전한 Point-in-Time 연구에서는 각 시점의 vintage 데이터를 사용해야 합니다.
"""
    )
    st.stop()

with st.spinner("데이터 수집 → feature 생성 → 모델 학습 → 예측 중..."):
    try:
        result = run_pipeline(
            market=market,
            target=target,
            horizon=horizon,
            train_years=train_years,
            settings=settings,
            min_train=min_train,
            calibrate=probability_bins,
        )
    except Exception as e:
        st.error("예측 실행에 실패했습니다.")
        st.exception(e)
        st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("현재 기준일", result["asof"].strftime("%Y-%m-%d"))
m2.metric("학습 표본", f'{result["n_train"]:,}')
m3.metric("예측 대상", result["target_label"])
m4.metric("데이터 상태", "LIVE" if result["live"] else "PARTIAL")

st.subheader(f"📊 {market} — {result['target_label']} — {horizon}개월")

probs = result["probabilities"]
cols = st.columns(3)
for col, label in zip(cols, ["상승", "횡보", "하락"]):
    col.metric(label, f"{probs.get(label, 0)*100:.1f}%")

st.bar_chart(pd.DataFrame({"확률": probs}).T)

left, right = st.columns(2)
with left:
    st.subheader("모델 진단")
    diag = result["diagnostics"]
    st.dataframe(pd.DataFrame([diag]), use_container_width=True, hide_index=True)
with right:
    st.subheader("현재 Macro Feature")
    current = result["current_features"].to_frame("현재값")
    st.dataframe(current, use_container_width=True)

st.subheader("종목별 예측")
stock_df = result["stock_predictions"].copy()
st.dataframe(
    stock_df.style.format({
        "상승확률": "{:.1%}",
        "횡보확률": "{:.1%}",
        "하락확률": "{:.1%}",
        "예상수익률 중앙값": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("최근 Walk-forward Backtest")
bt = result["backtest"]
if bt is not None and not bt.empty:
    st.dataframe(bt, use_container_width=True, hide_index=True)
    st.caption("Backtest는 과거 시점에서 미래를 예측하는 방식으로 계산합니다. 무작위 train/test split을 사용하지 않습니다.")
else:
    st.warning("백테스트 표본이 충분하지 않습니다.")

st.download_button(
    "현재 예측 결과 CSV",
    result["stock_predictions"].to_csv(index=False).encode("utf-8-sig"),
    file_name=f"{market}_{horizon}m_forecast.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "본 시스템은 통계적/머신러닝 모델의 불확실성을 포함합니다. 과거 성과는 미래 성과를 보장하지 않습니다. "
    "모델 오류, 데이터 지연·수정, 구조적 변화(regime shift), 거래비용 및 유동성은 별도로 고려해야 합니다."
)

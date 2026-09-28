import requests
import pandas as pd

BASE = "https://ecos.bok.or.kr/api/StatisticTableList"

def table_list(api_key, start=1, end=1000):
    if not api_key:
        raise ValueError("ECOS_API_KEY가 필요합니다.")
    url = f"{BASE}/{api_key}/json/kr/{start}/{end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def discover_table(api_key, keywords):
    """ECOS table list에서 이름에 keyword가 포함된 통계표 코드를 찾습니다."""
    data = table_list(api_key)
    rows = []
    # ECOS 응답은 환경/버전에 따라 StatisticTableList 안의 row에 들어갑니다.
    root = data.get("StatisticTableList", data)
    rows = root.get("row", []) if isinstance(root, dict) else []
    if isinstance(rows, dict):
        rows = [rows]
    for row in rows:
        name = str(row.get("STAT_NAME", "")) + " " + str(row.get("STAT_NAME_ENG", ""))
        if all(k.lower() in name.lower() for k in keywords):
            return row.get("STAT_CODE"), name
    return None, None

def statistic_search(api_key, stat_code, cycle, start, end):
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/10000/{stat_code}/{cycle}/{start}/{end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    root = data.get("StatisticSearch", data)
    rows = root.get("row", []) if isinstance(root, dict) else []
    if isinstance(rows, dict):
        rows = [rows]
    return pd.DataFrame(rows)

def get_metric(api_key, keywords, cycle="MM", start="201001", end=None):
    end = end or pd.Timestamp.today().strftime("%Y%m")
    code, name = discover_table(api_key, keywords)
    if not code:
        raise ValueError("ECOS 통계표 자동검색 실패: " + ",".join(keywords))
    df = statistic_search(api_key, code, cycle, start, end)
    if df.empty:
        raise ValueError(f"ECOS 데이터 없음: {name}")
    # 숫자형 값과 항목명 컬럼을 찾아 첫 번째 적합 항목을 선택
    value_col = "DATA_VALUE"
    if value_col not in df.columns:
        raise ValueError("ECOS 응답에 DATA_VALUE가 없습니다.")
    df[value_col] = pd.to_numeric(df[value_col].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=[value_col])
    item_name_col = "ITEM_NAME"
    if item_name_col in df.columns:
        # 가능한 경우 첫 번째 항목. 사용자는 config에서 더 정밀하게 지정할 수 있음.
        item = df[item_name_col].dropna().iloc[0]
        df = df[df[item_name_col] == item]
    time_col = "TIME"
    df["date"] = pd.to_datetime(df[time_col].astype(str), format="%Y%m", errors="coerce")
    return df.set_index("date")[value_col].rename(keywords[0]).sort_index()

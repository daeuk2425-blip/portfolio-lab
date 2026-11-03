import datetime as dt
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
import requests
from io import StringIO
import FinanceDataReader as fdr
from engine import simulate, analyze, FREQUENCIES
from report import make_pdf
from macro import ecos_key_statistics, kosis_statistics, interpret_macro
from research import ecos_series, kosis_series, transform, interpret_series, public_headlines
from narrative import write_analysis

st.set_page_config(page_title="Portfolio Lab | 나만의 투자 점검", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{max-width:1160px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.035em}
[data-testid="stMetric"]{background:#f4f7fb;border:1px solid #e3eaf3;border-radius:13px;padding:16px}
[data-testid="stSidebar"]{background:#f8fafc}
div.stButton>button[kind="primary"],div.stDownloadButton>button[kind="primary"]{border-radius:10px}
.hero{background:linear-gradient(125deg,#102b48,#1c5474);color:#fff;padding:32px;border-radius:18px;margin-bottom:22px}
.hero h1{color:white;font-size:2.2rem;margin:0 0 9px}.hero p{color:#dcecf5;margin:0}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>PORTFOLIO LAB</h1><p>구성하고 · 비교하고 · 위험을 이해하고 · 나만의 리서치 PDF를 받으세요.</p></div>',unsafe_allow_html=True)
st.caption("무료 분석 도구 | 원화 기준 백테스트 | 리밸런싱 전략 비교 | 검증 가능한 공식 통계 | 개인 투자 보고서")
st.info("**이용 순서** ① 종목 추가 및 비중 100% 맞추기 → ② 왼쪽에서 분석 조건 선택 → ③ 아래 '포트폴리오 진단 시작' → ④ 대시보드 점검 및 PDF 다운로드")

PRESETS = {
    "정기예금": "DEPOSIT", "RP형 CMA": "CMA", "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS", "현대자동차": "005380.KS", "NAVER": "035420.KS",
    "카카오": "035720.KS", "삼성전자우": "005935.KS", "애플": "AAPL",
    "엔비디아": "NVDA", "마이크로소프트": "MSFT", "테슬라": "TSLA",
    "아마존": "AMZN", "알파벳": "GOOGL", "메타": "META",
    "비트코인": "BTC-USD", "이더리움": "ETH-USD", "S&P 500 ETF (SPY)": "SPY",
    "나스닥 100 ETF (QQQ)": "QQQ", "금 ETF (GLD)": "GLD", "미국 국채 ETF (TLT)": "TLT",
}
ALIASES = {k.replace(" ", "").upper(): v for k, v in PRESETS.items()}
ALIASES.update({"APPLE":"AAPL", "NVIDIA":"NVDA", "BITCOIN":"BTC-USD", "ETHEREUM":"ETH-USD", "SAMSUNG":"005930.KS"})
DEFAULT = pd.DataFrame({"종목명": ["정기예금", "RP형 CMA", "비트코인", "S&P 500 ETF (SPY)"], "종목 코드": ["DEPOSIT", "CMA", "BTC-USD", "SPY"], "비중(%)": [30., 20., 30., 20.]})

def normalize_ticker(value):
    text = str(value).strip()
    return ALIASES.get(text.replace(" ", "").upper(), text.upper())

@st.cache_data(ttl=3600, show_spinner=False)
def search_yahoo(query):
    try:
        results = yf.Search(query, max_results=8).quotes
        out = []
        for item in results:
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                continue
            # Other quote currencies need an explicit FX conversion before analysis.
            if not (symbol.endswith((".KS", ".KQ")) or item.get("currency") in ("USD", "KRW")):
                continue
            label = item.get("shortname") or item.get("longname") or symbol
            out.append((f"{label} — {symbol}", symbol))
        return out
    except Exception:
        return []


def is_krw_ticker(ticker):
    return ticker.endswith((".KS", ".KQ"))

@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError("가격 데이터를 다운로드하지 못했습니다.")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.sort_index().resample("ME").last()

def metrics(r, market, rf):
    years = len(r) / 12
    growth = (1 + r).cumprod()
    total = growth.iloc[-1] - 1
    cagr = growth.iloc[-1] ** (1 / years) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    rf_month = (1 + rf) ** (1 / 12) - 1
    sharpe = (r - rf_month).mean() / r.std(ddof=1) * np.sqrt(12) if r.std(ddof=1) > 0 else np.nan
    mdd = (growth / growth.cummax() - 1).min()
    beta = r.cov(market) / market.var(ddof=1) if market.var(ddof=1) > 0 else np.nan
    # Monthly CAPM alpha, annualized geometrically; market and portfolio are in KRW.
    alpha_month = (r - rf_month).mean() - beta * (market - rf_month).mean() if np.isfinite(beta) else np.nan
    alpha_annual = (1 + alpha_month) ** 12 - 1 if np.isfinite(alpha_month) and alpha_month > -1 else np.nan
    return total, cagr, vol, sharpe, mdd, beta, alpha_annual

st.sidebar.header("분석 설정")
years = st.sidebar.selectbox("분석 기간", [1, 3, 5, 10], index=3)
deposit_rate = st.sidebar.number_input("정기예금 가정 연 금리 (%)", min_value=0., max_value=30., value=3., step=0.1)
cma_rate = st.sidebar.number_input("RP형 CMA 가정 연 금리 (%)", min_value=0., max_value=30., value=3., step=0.1)
rf_rate = st.sidebar.number_input("무위험수익률 가정 연 금리 (%)", min_value=0., max_value=30., value=3., step=0.1)
st.sidebar.caption("예금·CMA·무위험수익률은 실제 과거 금리 자료가 아닌 사용자 가정값입니다.")
strategy = st.sidebar.selectbox("리밸런싱 주기", list(FREQUENCIES), index=1)
transaction_bps = st.sidebar.number_input("리밸런싱 거래비용 (bps, 편도)", min_value=0.0, max_value=500.0, value=10.0, step=5.0)
st.sidebar.caption("거래비용은 매매 회전율에 비례해 리밸런싱 시점에 반영합니다. 세금·슬리피지는 별도입니다.")

# Listings are fetched on demand, cached for 24 hours, and may occasionally be unavailable.
# A searchable universe is not a guarantee of historical price availability.
@st.cache_data(ttl=86400, show_spinner=False)
def load_kospi():
    df = fdr.StockListing("KOSPI")
    if df is None or df.empty:
        raise RuntimeError("KRX 종목 목록이 비어 있습니다")
    code_col = next((c for c in ("Code", "Symbol", "Ticker") if c in df.columns), None)
    name_col = next((c for c in ("Name", "Company") if c in df.columns), None)
    if not code_col or not name_col:
        raise RuntimeError("KRX 종목 목록의 열 구성이 변경되었습니다")
    result = pd.DataFrame({"종목명": df[name_col].astype(str),
                           "티커": df[code_col].astype(str).str.zfill(6) + ".KS",
                           "시장": "KOSPI"})
    return result.drop_duplicates("티커")

@st.cache_data(ttl=86400, show_spinner=False)
def load_nasdaq():
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep="|", dtype=str)
    df = df[df["Symbol"].notna() & (df["Symbol"] != "File Creation Time")]
    if "Test Issue" in df:
        df = df[df["Test Issue"] == "N"]
    # Yahoo Finance generally uses '-' for NASDAQ symbols containing a dot.
    result = pd.DataFrame({"종목명": df["Security Name"].astype(str),
                           "티커": df["Symbol"].astype(str).str.replace(".", "-", regex=False),
                           "시장": "NASDAQ"})
    return result.drop_duplicates("티커")

@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = next((t for t in tables if "Symbol" in t.columns and "Security" in t.columns), None)
    if df is None:
        raise RuntimeError("S&P 500 구성종목 표를 찾지 못했습니다")
    result = pd.DataFrame({"종목명": df["Security"].astype(str),
                           "티커": df["Symbol"].astype(str).str.replace(".", "-", regex=False),
                           "시장": "S&P 500"})
    return result.drop_duplicates("티커")

@st.cache_data(ttl=86400, show_spinner=False)
def load_crypto():
    return pd.DataFrame([
        {"종목명": "비트코인 (Bitcoin)", "티커": "BTC-USD", "시장": "암호화폐"},
        {"종목명": "이더리움 (Ethereum)", "티커": "ETH-USD", "시장": "암호화폐"},
    ])

st.subheader("1. 전체 시장 종목 검색 및 포트폴리오 구성")
st.caption("코스피·나스닥 상장종목, S&P 500 지수 구성종목, 비트코인·이더리움을 검색할 수 있습니다. 상장종목 목록과 가격 데이터는 서로 다른 출처입니다.")
if "portfolio_v7" not in st.session_state:
    # Start clean so the table never disagrees with hidden editor state.
    st.session_state.portfolio_v7 = DEFAULT.to_dict("records")

market_choice = st.radio("검색 대상", ["코스피", "나스닥", "S&P 500", "암호화폐 (BTC·ETH)", "기타 티커 직접 입력"], horizontal=True)
query = st.text_input("종목명 또는 티커 검색", placeholder="예: LG에너지솔루션, Apple, NVDA")

catalog = pd.DataFrame(columns=["종목명", "티커", "시장"])
if market_choice != "기타 티커 직접 입력":
    try:
        with st.spinner("종목 목록을 불러오는 중..."):
            catalog = {"코스피": load_kospi, "나스닥": load_nasdaq, "S&P 500": load_sp500, "암호화폐 (BTC·ETH)": load_crypto}[market_choice]()
        st.caption(f"{market_choice} 검색 대상: {len(catalog):,}개")
    except Exception as exc:
        st.warning(f"종목 목록 조회 실패: {exc}. 직접 티커를 입력할 수 있습니다.")

matches = pd.DataFrame(columns=["종목명", "티커", "시장"])
if not catalog.empty:
    if query.strip():
        mask = (catalog["종목명"].str.contains(query.strip(), case=False, regex=False, na=False)
                | catalog["티커"].str.contains(query.strip(), case=False, regex=False, na=False))
        matches = catalog.loc[mask].head(60)
    else:
        st.caption("종목명을 검색하면 결과가 나타납니다.")

options = []
if not matches.empty:
    options = [(str(r["종목명"]), str(r["티커"])) for _, r in matches.iterrows()]
    st.dataframe(matches.head(15), hide_index=True, use_container_width=True)

with st.form("add_asset_form", clear_on_submit=False):
    selected = st.selectbox("검색 결과에서 종목 선택", options,
                            format_func=lambda x: f"{x[0]} · {x[1]}",
                            disabled=not options, placeholder="검색 결과에서 선택")
    manual_code = st.text_input("또는 티커 직접 입력", placeholder="예: AAPL, 373220.KS, BTC-USD, DEPOSIT")
    add_weight = st.number_input("새 종목 비중 (%)", 0.0, 100.0, 10.0, 1.0)
    submitted = st.form_submit_button("＋ 포트폴리오에 추가", type="primary")

if submitted:
    if manual_code.strip():
        code = normalize_ticker(manual_code)
        display_name = next((n for n, t in PRESETS.items() if t == code), code)
        if not catalog.empty:
            hit = catalog[catalog["종목명"].str.casefold() == manual_code.strip().casefold()]
            if len(hit) == 1:
                code, display_name = str(hit.iloc[0]["티커"]), str(hit.iloc[0]["종목명"])
    elif selected:
        display_name, code = selected
    else:
        code, display_name = "", ""
    if not code:
        st.warning("먼저 종목을 검색해 선택하거나 티커를 입력해 주세요.")
    elif any(x["종목 코드"] == code for x in st.session_state.portfolio_v7):
        st.info(f"{display_name}은 이미 있습니다. 아래에서 비중을 변경할 수 있어요.")
    elif not (code in ("DEPOSIT", "CMA") or code.endswith((".KS", ".KQ")) or code.isascii()):
        st.warning("한글 종목명은 검색 결과에서 선택해 주세요.")
    else:
        st.session_state.portfolio_v7.append({"종목명": display_name, "종목 코드": code, "비중(%)": float(add_weight)})
        st.rerun()

st.subheader("현재 포트폴리오")
st.caption("종목별 비중을 바로 수정하거나 삭제할 수 있습니다. 비중을 바꾸면 즉시 합계가 갱신됩니다.")

# Stable ticker-specific widget keys avoid data_editor's stale-row state.
for asset in list(st.session_state.portfolio_v7):
    code = asset["종목 코드"]
    c1, c2, c3 = st.columns([5, 2, 1], vertical_alignment="center")
    c1.markdown(f"**{asset['종목명']}**  \n`{code}`")
    weight_key = "weight_v7_" + code
    if weight_key not in st.session_state:
        st.session_state[weight_key] = float(asset["비중(%)"])
    new_weight = c2.number_input("비중 (%)", min_value=0.0, max_value=100.0,
                                  step=1.0, key=weight_key, label_visibility="collapsed")
    asset["비중(%)"] = float(new_weight)
    if c3.button("삭제", key="remove_v7_" + code):
        st.session_state.portfolio_v7 = [x for x in st.session_state.portfolio_v7 if x["종목 코드"] != code]
        st.session_state.pop(weight_key, None)
        st.rerun()

allocation = pd.DataFrame(st.session_state.portfolio_v7, columns=["종목명", "종목 코드", "비중(%)"])
total_weight = float(allocation["비중(%)"].sum()) if not allocation.empty else 0.0
st.progress(min(total_weight / 100.0, 1.0), text=f"현재 비중 합계: {total_weight:.1f}% / 100%")

c_norm, c_reset = st.columns(2)
if c_norm.button("비중 자동으로 100% 맞추기", disabled=allocation.empty or total_weight <= 0):
    for asset in st.session_state.portfolio_v7:
        adjusted = float(asset["비중(%)"]) / total_weight * 100
        asset["비중(%)"] = adjusted
        st.session_state["weight_v7_" + asset["종목 코드"]] = adjusted
    st.rerun()
if c_reset.button("포트폴리오 초기화"):
    for asset in st.session_state.portfolio_v7:
        st.session_state.pop("weight_v7_" + asset["종목 코드"], None)
    st.session_state.portfolio_v7 = []
    st.rerun()

if not np.isclose(total_weight, 100):
    st.warning("분석하려면 비중 합계를 100%로 맞춰 주세요. 자동 조정 버튼을 이용할 수 있습니다.")
st.caption("예금·CMA 금리는 가정값이며, 국내 주식은 KRW, 그 외에는 USD 표시 자산만 지원합니다.")

st.subheader("공식 통계 · 경제 뉴스 (선택 설정)")
st.caption("인증키가 없어도 포트폴리오 백테스트와 PDF를 생성할 수 있습니다.")
st.caption("AI 유료 API를 호출하지 않습니다. 통계 인증키는 공식 기관에서 무료 신청할 수 있으며 이용조건을 확인해야 합니다.")
ecos_key = st.text_input("한국은행 ECOS 인증키 (선택)", type="password")
kosis_key = st.text_input("KOSIS 인증키 (선택)", type="password")
kosis_org = st.text_input("KOSIS 기관코드 (선택)", placeholder="공식 통계표에 표시된 orgId")
kosis_table = st.text_input("KOSIS 통계표코드 (선택)", placeholder="공식 통계표에 표시된 tblId")
st.caption("인증키를 ZIP이나 공개 코드에 저장하지 마세요. 공식 통계표의 정확한 항목코드를 사용해야 합니다.")
st.markdown("#### 시계열 원자료 분석 설정")
macro_provider = st.selectbox("시계열 출처", ["ECOS", "KOSIS"])
macro_name = st.text_input("지표 표시명", value="선택한 경제지표", max_chars=60)
ecos_table = st.text_input("ECOS 통계표코드", placeholder="ECOS 통계표 검색에서 확인")
ecos_item = st.text_input("ECOS 항목코드", placeholder="해당 통계표의 ITEM_CODE1")
kosis_item = st.text_input("KOSIS 항목코드", value="ALL")
kosis_obj = st.text_input("KOSIS 분류코드", value="ALL")
macro_transform = st.selectbox("데이터 가공", ["원자료", "전년동월비 (%)", "전월비 (%)", "3개월 이동평균"])
news_query = st.text_input("경제신문 뉴스 검색어", value="한국 금리 환율 경기", max_chars=100)
st.caption("뉴스는 RSS 제목과 공개 메타데이터만 수집합니다. 기사 전문이나 증권사 리서치 PDF를 읽었다고 주장하지 않습니다.")

if st.button("📊 포트폴리오 진단 시작 · PDF 만들기", type="primary", use_container_width=True):
    try:
        if allocation.empty or allocation.isna().any().any():
            raise ValueError("자산과 비중을 모두 입력해 주세요.")
        names = [str(v).strip().upper() for v in allocation["종목 코드"].tolist()]
        if len(names) != len(set(names)) or any(not x for x in names):
            raise ValueError("종목 코드가 비어 있거나 중복되었습니다.")
        weights = pd.Series(allocation["비중(%)"].to_numpy(dtype=float) / 100, index=names)
        if not np.isclose(weights.sum(), 1., atol=1e-6):
            raise ValueError("비중 합계가 100%가 되어야 합니다.")
        end = dt.date.today() + dt.timedelta(days=1)
        start = dt.date.today() - dt.timedelta(days=round(years * 365.25) + 65)
        selected_tickers = [x for x in names if x not in ("DEPOSIT", "CMA")]
        tickers = tuple(sorted(set(selected_tickers + ["SPY", "KRW=X"])))
        with st.spinner("가격과 환율 데이터를 다운로드하는 중..."):
            prices = get_prices(tickers, start.isoformat(), end.isoformat())
        missing = [x for x in tickers if x not in prices or prices[x].dropna().empty]
        if missing:
            raise ValueError("가격 데이터가 없는 종목: " + ", ".join(missing))
        prices = prices[list(tickers)].ffill().dropna()
        prices = prices[prices.index.to_period("M") < pd.Timestamp.today().to_period("M")]
        prices = prices.tail(years * 12 + 1)
        if len(prices) < 7:
            raise ValueError("공통 가격 이력이 부족합니다. 기간을 줄여 주세요.")
        usdkrw = prices["KRW=X"]
        benchmark = (prices["SPY"] * usdkrw).pct_change().dropna()
        monthly = pd.DataFrame(index=benchmark.index)
        for ticker in names:
            if ticker in ("DEPOSIT", "CMA"):
                rate = deposit_rate / 100 if ticker == "DEPOSIT" else cma_rate / 100
                monthly[ticker] = (1 + rate) ** (1 / 12) - 1
            else:
                price_krw = prices[ticker] if is_krw_ticker(ticker) else prices[ticker] * usdkrw
                monthly[ticker] = price_krw.pct_change().reindex(benchmark.index)
        combined = monthly.join(benchmark.rename("benchmark"), how="inner").dropna()
        monthly, benchmark = combined[names], combined["benchmark"]
        if len(monthly) < 6:
            raise ValueError("공통 수익률 이력이 6개월 미만입니다.")
        all_stats, all_returns, all_turnovers = {}, {}, {}
        for label, interval in FREQUENCIES.items():
            r, turnover = simulate(monthly, weights, interval, transaction_bps)
            all_returns[label] = r
            all_turnovers[label] = turnover
            all_stats[label] = analyze(r, benchmark, rf_rate / 100)
        stats = all_stats[strategy]
        port = all_returns[strategy]
        st.success("분석이 완료되었습니다. 아래에서 포트폴리오의 성과·위험·리밸런싱 결과를 확인하세요.")
        st.subheader("핵심 성과 대시보드")
        percent_keys = {"누적 수익률", "CAGR", "벤치마크 CAGR", "벤치마크 대비 CAGR 차이", "연환산 변동성", "Jensen Alpha (연율 근사)", "Treynor (연율 근사)", "MDD", "VaR 95% (월간, 역사적)", "VaR 99% (월간, 역사적)"}
        for i in range(0, len(stats), 4):
            cols = st.columns(4)
            for col, (key, value) in zip(cols, list(stats.items())[i:i+4]):
                display = "산출 불가" if not np.isfinite(value) else (f"{value:.2%}" if key in percent_keys else f"{value:.3f}")
                col.metric(key, display)
        comparison = pd.DataFrame(all_stats).T
        st.subheader("리밸런싱 전략별 성과 비교")
        st.dataframe(comparison[["CAGR", "MDD", "Sharpe", "연환산 변동성"]].style.format({"CAGR":"{:.2%}","MDD":"{:.2%}","Sharpe":"{:.3f}","연환산 변동성":"{:.2%}"}), use_container_width=True)
        growth = pd.DataFrame({label:(1+r).cumprod() for label,r in all_returns.items()})
        growth["SPY 벤치마크 (KRW)"] = (1+benchmark).cumprod()
        st.plotly_chart(px.line(growth, title="리밸런싱 주기별 누적 성장 (초기 1 기준)"), use_container_width=True)
        dd = (1+port).cumprod()
        dd = dd / dd.cummax() - 1
        st.plotly_chart(px.line(dd, title=f"{strategy} 포트폴리오 낙폭"), use_container_width=True)
        corr = monthly.corr()
        st.subheader("자산별 상관관계")
        st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r"), use_container_width=True)
        st.caption(f"공통 분석기간: {monthly.index.min():%Y-%m} ~ {monthly.index.max():%Y-%m} ({len(monthly)}개월). 99% VaR는 짧은 표본에서 불안정할 수 있습니다.")
        st.caption("월말 데이터 · KRW 환산 · SPY 벤치마크 · 과거 가격 · 예금/CMA 고정 가정금리 · 세금 미반영")
        export = monthly.copy()
        export["포트폴리오"] = port
        export["SPY 벤치마크"] = benchmark
        st.download_button("월별 수익률 CSV", export.to_csv(index_label="월").encode("utf-8-sig"), "portfolio_monthly_returns.csv", "text/csv")
        with st.spinner("공식 통계 조회 및 포트폴리오 위험 해석 중..."):
            ecos_obs, ecos_error = ecos_key_statistics(ecos_key)
            kosis_obs, kosis_error = kosis_statistics(kosis_key, kosis_org, kosis_table)
        observations = ecos_obs + kosis_obs
        if ecos_error: st.info(ecos_error)
        if kosis_error: st.info(kosis_error)
        st.subheader("공식 통계 관측치")
        if observations:
            st.dataframe(pd.DataFrame(observations)[['source','name','value','unit','period']], use_container_width=True)
        else:
            st.warning("공식 통계 관측치를 확인하지 못했습니다. 현재 경제 상황에 대한 수치 주장은 작성하지 않습니다.")
        macro_view = interpret_macro(allocation, stats, observations)
        macro_series = pd.Series(dtype=float)
        if macro_provider == "ECOS" and ecos_key and ecos_table and ecos_item:
            macro_series, series_error = ecos_series(ecos_key, ecos_table, ecos_item)
        elif macro_provider == "KOSIS" and kosis_key and kosis_org and kosis_table:
            macro_series, series_error = kosis_series(kosis_key, kosis_org, kosis_table, kosis_item, kosis_obj)
        else:
            series_error = "시계열 데이터 설정이 불완전해 차트를 생략했습니다."
        if series_error: st.info(series_error)
        macro_chart = None
        if not macro_series.empty:
            macro_chart = transform(macro_series, macro_transform).dropna()
            if len(macro_chart) >= 2:
                st.subheader("원자료 가공 시계열")
                st.line_chart(macro_chart.rename(macro_name))
                st.download_button("가공된 경제통계 CSV", macro_chart.rename(macro_name).to_csv(index_label="date").encode("utf-8-sig"), "macro_timeseries.csv", "text/csv")
                macro_view = interpret_series(macro_chart, macro_name, macro_transform, float(allocation.loc[~allocation['종목 코드'].str.endswith(('.KS','.KQ')) & ~allocation['종목 코드'].isin(['DEPOSIT','CMA']), '비중(%)'].sum())) + " " + macro_view
            else:
                macro_chart = None
        try:
            headlines = public_headlines(news_query) if news_query.strip() else []
        except Exception as news_error:
            headlines = []
            st.info("뉴스 메타데이터 조회 실패: " + str(news_error)[:100])
        if headlines:
            st.subheader("관련 뉴스 제목 (기사 본문 미확인)")
            for item in headlines: st.markdown(f"- [{item['title']}]({item['url']}) · {item['published']}")
        st.subheader("포트폴리오 관점의 거시경제 해석 (규칙 기반)")
        st.write(macro_view)
        sources = [(f"{o['source']} | {o['name']} | {o['period']}",o['url']) for o in observations[:4]]
        if macro_chart is not None:
            sources.insert(0,(f"{macro_provider} | {macro_name} | {macro_chart.index[-1]:%Y-%m} | {macro_transform}","https://ecos.bok.or.kr/" if macro_provider=="ECOS" else "https://kosis.kr/"))
        sources += [(f"뉴스 제목만 확인: {h['title']} ({h['published']})",h['url']) for h in headlines[:3]]
        analyst = write_analysis(allocation, stats, comparison, observations, macro_view, len(monthly))
        st.subheader("포트폴리오 핵심 진단")
        st.write(analyst["thesis"])
        st.write(analyst["risk_interpretation"])
        st.write(analyst["conclusion"])
        assumptions = ["월말 종가 기준, 완료된 월만 사용, 분석 가능한 공통 기간으로 제한", "USD 표시 자산은 USD/KRW로 원화 환산", f"리밸런싱: {strategy}, 편도 거래비용: {transaction_bps:.0f} bps (회전율 기준)", f"예금: {deposit_rate:.2f}%, CMA: {cma_rate:.2f}%, 무위험수익률: {rf_rate:.2f}% (연 고정 가정)", "실제 거래, 세금, 매매 스프레드, 배당 원천징수, 예금 중도해지 등 미반영", "선택 종목의 과거 현재 생존 종목 기준 분석으로 생존편향 가능"]
        try:
            pdf = make_pdf(allocation, strategy, stats, comparison, monthly, port, benchmark, corr, sources, assumptions, analyst=analyst, macro_view=macro_view, observations=observations, macro_chart=macro_chart, macro_label=macro_name, macro_transform=macro_transform)
            st.download_button("📄 4페이지 개인 투자 분석 PDF 다운로드", data=pdf, file_name="portfolio_lab_report.pdf", mime="application/pdf", type="primary")
        except Exception as pdf_exc:
            st.error(f"PDF 생성 실패: {pdf_exc}")
            st.info("Colab에서 !apt-get -qq install fonts-nanum 실행 후 다시 분석해 주세요.")
        st.info("과거 시뮬레이션이며 투자 권유가 아닙니다. 분석 문단은 검증된 계산값과 조건부 규칙으로 생성되며, 생성형 AI가 직접 조사·집필한 의견은 아닙니다.")
    except Exception as exc:
        st.error(f"분석을 완료하지 못했습니다: {exc}")

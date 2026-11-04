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
st.caption("무료 분석 도구 | 원화 기준 백테스트 | 리밸런싱 전략 비교 | 개인 투자 보고서")
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
def load_krx(market):
    """Daily cached Korean listing, including listed ETFs."""
    df = fdr.StockListing(market)
    if df is None or df.empty:
        raise RuntimeError(f"{market} 종목 목록이 비어 있습니다")
    code_col = next((c for c in ("Code", "Symbol", "Ticker") if c in df.columns), None)
    name_col = next((c for c in ("Name", "Company") if c in df.columns), None)
    if not code_col or not name_col:
        raise RuntimeError(f"{market} 종목 목록의 열 구성이 변경되었습니다")
    suffix = ".KQ" if market == "KOSDAQ" else ".KS"
    return pd.DataFrame({"종목명": df[name_col].astype(str),
                         "티커": df[code_col].astype(str).str.zfill(6) + suffix,
                         "시장": market}).drop_duplicates("티커")

@st.cache_data(ttl=86400, show_spinner=False)
def load_nasdaq():
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep="|", dtype=str)
    df = df[df["Symbol"].notna() & (df["Symbol"] != "File Creation Time")]
    if "Test Issue" in df:
        df = df[df["Test Issue"] == "N"]
    return pd.DataFrame({"종목명": df["Security Name"].astype(str),
                         "티커": df["Symbol"].astype(str).str.replace(".", "-", regex=False),
                         "시장": "NASDAQ"}).drop_duplicates("티커")

@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = next((t for t in tables if "Symbol" in t.columns and "Security" in t.columns), None)
    if df is None:
        raise RuntimeError("S&P 500 구성종목 표를 찾지 못했습니다")
    return pd.DataFrame({"종목명": df["Security"].astype(str),
                         "티커": df["Symbol"].astype(str).str.replace(".", "-", regex=False),
                         "시장": "S&P 500"}).drop_duplicates("티커")

@st.cache_data(ttl=86400, show_spinner=False)
def load_crypto():
    return pd.DataFrame([
        {"종목명": "비트코인 (Bitcoin)", "티커": "BTC-USD", "시장": "암호화폐"},
        {"종목명": "이더리움 (Ethereum)", "티커": "ETH-USD", "시장": "암호화폐"},
    ])

US_KOREAN_NAMES = {
    "AAPL": "애플", "NVDA": "엔비디아", "MSFT": "마이크로소프트",
    "TSLA": "테슬라", "AMZN": "아마존", "GOOGL": "알파벳 구글",
    "GOOG": "구글", "META": "메타 페이스북", "NFLX": "넷플릭스",
    "AMD": "AMD 에이엠디", "PLTR": "팔란티어", "COIN": "코인베이스",
    "SPY": "S&P500 에스앤피", "QQQ": "나스닥100",
}

# Frequently searched ETFs remain searchable even if an external listing provider fails.
ETF_FALLBACK = [
    ("SPDR S&P 500 ETF Trust (SPY)", "SPY", "미국 ETF", "에스앤피500 S&P500"),
    ("Invesco QQQ Trust (QQQ)", "QQQ", "미국 ETF", "나스닥100"),
    ("Vanguard S&P 500 ETF (VOO)", "VOO", "미국 ETF", "에스앤피500 S&P500"),
    ("Vanguard Total Stock Market ETF (VTI)", "VTI", "미국 ETF", "미국전체시장"),
    ("Schwab US Dividend Equity ETF (SCHD)", "SCHD", "미국 ETF", "미국배당"),
    ("iShares Core S&P 500 ETF (IVV)", "IVV", "미국 ETF", "에스앤피500 S&P500"),
    ("Invesco Nasdaq 100 ETF (QQQM)", "QQQM", "미국 ETF", "나스닥100"),
    ("iShares 20+ Year Treasury Bond ETF (TLT)", "TLT", "미국 ETF", "미국장기국채"),
    ("SPDR Gold Shares (GLD)", "GLD", "미국 ETF", "금 골드"),
    ("KODEX 200", "069500.KS", "국내 ETF", "코덱스 코스피200"),
    ("TIGER 200", "102110.KS", "국내 ETF", "타이거 코스피200"),
    ("KODEX 코스닥150", "229200.KS", "국내 ETF", "코덱스 코스닥150"),
    ("TIGER 미국S&P500", "360750.KS", "국내 ETF", "타이거 미국 에스앤피500"),
    ("KODEX 미국S&P500", "379800.KS", "국내 ETF", "코덱스 미국 에스앤피500"),
    ("TIGER 미국나스닥100", "133690.KS", "국내 ETF", "타이거 미국 나스닥100"),
]

@st.cache_data(ttl=86400, show_spinner=False)
def load_krx_etfs():
    errors = []
    for market in ("KRX-ETF", "ETF"):
        try:
            return load_krx(market)
        except Exception as exc:
            errors.append(f"{market}: {exc}")
    raise RuntimeError(" / ".join(errors))

@st.cache_data(ttl=86400, show_spinner=False)
def load_catalog():
    frames, errors = [], []
    for label, loader in [("KOSPI", lambda: load_krx("KOSPI")),
                          ("KOSDAQ", lambda: load_krx("KOSDAQ")),
                          ("KRX ETF", load_krx_etfs),
                          ("NASDAQ", load_nasdaq), ("S&P 500", load_sp500),
                          ("암호화폐", load_crypto)]:
        try:
            frames.append(loader())
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    fallback = pd.DataFrame(ETF_FALLBACK, columns=["종목명", "티커", "시장", "별칭"])
    catalog = pd.concat([fallback, *frames], ignore_index=True).drop_duplicates("티커", keep="first")
    catalog["검색어"] = (catalog["종목명"].astype(str) + " " + catalog["티커"].astype(str)
                          + " " + catalog["티커"].map(US_KOREAN_NAMES).fillna("")
                          + " " + catalog.get("별칭", pd.Series(index=catalog.index, dtype=str)).fillna(""))
    return catalog, errors

st.subheader("1. 통합 종목 검색 및 포트폴리오 구성")
st.caption("코스피·코스닥·국내 ETF·나스닥·S&P500·암호화폐를 한 번에 검색합니다. 종목 목록은 하루 동안 캐시됩니다.")
if "portfolio_v7" not in st.session_state:
    st.session_state.portfolio_v7 = DEFAULT.to_dict("records")
query = st.text_input("종목명·한글명·영문명·티커 통합 검색", placeholder="예: TIGER 반도체, 애플, Apple, NVDA, 069500")
with st.spinner("통합 종목 목록을 불러오는 중..."):
    catalog, listing_errors = load_catalog()
if listing_errors:
    st.caption("일부 목록을 불러오지 못했습니다: " + " | ".join(listing_errors)[:300])
st.caption(f"검색 가능한 종목: {len(catalog):,}개 (중복 티커 제거)")
matches = pd.DataFrame(columns=["종목명", "티커", "시장"])
if query.strip() and not catalog.empty:
    q = query.strip().casefold().replace(" ", "")
    mask = catalog["검색어"].str.casefold().str.replace(" ", "", regex=False).str.contains(q, regex=False, na=False)
    matches = catalog.loc[mask, ["종목명", "티커", "시장"]].head(60)
    # Yahoo search supplements, but does not replace, the local searchable catalog.
    # Search only for Latin-letter queries to avoid unreliable Korean transliteration.
    if len(matches) < 15 and any("a" <= ch.lower() <= "z" for ch in query):
        yahoo_rows = []
        for label, ticker in search_yahoo(query.strip()):
            yahoo_rows.append({"종목명": label, "티커": ticker, "시장": "Yahoo 검색"})
        if yahoo_rows:
            matches = pd.concat([matches, pd.DataFrame(yahoo_rows)], ignore_index=True).drop_duplicates("티커").head(60)
elif not query.strip():
    st.caption("검색어를 입력하면 통합 결과가 나타납니다.")

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

st.caption("경제통계 API(ECOS·KOSIS)는 사용하지 않습니다. 결과는 과거 가격과 사용자 가정에 기반합니다.")

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
        tickers = tuple(sorted(set(selected_tickers + ["SPY", "069500.KS", "KRW=X"])))
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
        kospi200_benchmark = prices["069500.KS"].pct_change().dropna()  # KODEX 200 ETF proxy
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
        kospi200_benchmark = kospi200_benchmark.reindex(monthly.index)
        valid = kospi200_benchmark.notna()
        monthly, benchmark, kospi200_benchmark = monthly.loc[valid], benchmark.loc[valid], kospi200_benchmark.loc[valid]
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
        growth["S&P500 (SPY, KRW)"] = (1+benchmark).cumprod()
        growth["KOSPI200 (KODEX 200 ETF 대용)"] = (1+kospi200_benchmark).cumprod()
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
        from engine import analyze_benchmarks
        dual_stats = analyze_benchmarks(port, pd.DataFrame({"S&P500 (SPY, KRW)": benchmark, "KOSPI200 (KODEX 200 ETF)": kospi200_benchmark}), rf_rate / 100)
        st.subheader("벤치마크별 위험조정성과")
        st.dataframe(pd.DataFrame(dual_stats).T, use_container_width=True)
        st.caption("KOSPI200은 KODEX 200 ETF(069500.KS), S&P500은 SPY ETF를 사용한 대용 비교입니다. 지수 자체의 총수익률과 다를 수 있습니다.")
        observations, macro_view, sources = [], "경제통계 API는 사용하지 않습니다.", []
        analyst = write_analysis(allocation, stats, comparison, observations, macro_view, len(monthly))
        st.subheader("포트폴리오 핵심 진단")
        for key in ("thesis", "risk_interpretation", "conclusion"):
            if analyst.get(key):
                st.write(analyst[key])
        assumptions = ["월말 종가 기준, 완료된 월만 사용, 분석 가능한 공통 기간으로 제한", "USD 표시 자산은 USD/KRW로 원화 환산", f"리밸런싱: {strategy}, 편도 거래비용: {transaction_bps:.0f} bps", f"예금: {deposit_rate:.2f}%, CMA: {cma_rate:.2f}%, 무위험수익률: {rf_rate:.2f}% (연 고정 가정)", "실제 거래·세금·매매 스프레드 미반영", "과거 생존 종목 중심 분석으로 생존편향 가능"]
        try:
            pdf = make_pdf(allocation, strategy, stats, comparison, monthly, port, benchmark, corr, sources, assumptions, analyst=analyst, macro_view=macro_view, observations=observations, benchmark_kospi200=kospi200_benchmark, benchmark_stats=dual_stats)
            st.download_button("📄 종합 투자 분석 PDF 다운로드", data=pdf, file_name="portfolio_lab_report.pdf", mime="application/pdf", type="primary")
        except Exception as pdf_exc:
            st.error(f"PDF 생성 실패: {pdf_exc}")
            st.caption("report.py와 한글 글꼴 설치 상태를 확인해 주세요.")
        st.info("과거 시뮬레이션이며 투자 권유가 아닙니다. 분석 문단은 검증된 계산값과 조건부 규칙으로 생성되며, 생성형 AI가 직접 조사·집필한 의견은 아닙니다.")
    except Exception as exc:
        st.error(f"분석을 완료하지 못했습니다: {exc}")
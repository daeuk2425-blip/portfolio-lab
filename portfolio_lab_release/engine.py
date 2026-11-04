"""Portfolio Lab risk engine v2. Monthly return inputs, decimal units."""
import numpy as np
import pandas as pd

FREQUENCIES = {'리밸런싱 없음': None, '연 1회': 12, '반기 1회': 6, '분기 1회': 3, '월 1회': 1}

def simulate(monthly, weights, interval=None, cost_bps=0.0):
    """End-of-month rebalance; turnover is one-way fraction of portfolio."""
    w0 = weights.reindex(monthly.columns).astype(float).to_numpy()
    if not np.isfinite(w0).all() or not np.isclose(w0.sum(), 1):
        raise ValueError('비중 합계는 100%여야 합니다.')
    holdings = w0.copy()
    results, turnovers = [], []
    for i, (_, row) in enumerate(monthly.iterrows()):
        old = holdings.sum()
        holdings = holdings * (1 + row.to_numpy(dtype=float))
        gross = holdings.sum() / old - 1
        fee, turnover = 0.0, 0.0
        if interval is not None and (i + 1) % interval == 0 and i + 1 < len(monthly):
            target = holdings.sum() * w0
            turnover = np.abs(target - holdings).sum() / (2 * holdings.sum())
            fee = turnover * cost_bps / 10000
            holdings = target * (1 - fee)
        turnovers.append(turnover)
        results.append((1 + gross) * (1 - fee) - 1)
    return pd.Series(results, index=monthly.index, name='포트폴리오'), pd.Series(turnovers, index=monthly.index)

def analyze(r, benchmark, rf_annual, confidence=0.95):
    """Backward-compatible single benchmark analysis. Monthly KRW returns."""
    r, benchmark = r.align(benchmark, join='inner')
    pair = pd.concat([r.rename('p'), benchmark.rename('b')], axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    r, benchmark = pair['p'], pair['b']
    if len(r) < 6:
        raise ValueError('위험지표 계산을 위해 최소 6개월의 공통 데이터가 필요합니다.')
    if (r <= -1).any() or (benchmark <= -1).any():
        raise ValueError('월간 수익률은 -100%보다 커야 합니다.')
    growth = (1+r).cumprod()
    bgrowth = (1+benchmark).cumprod()
    years = len(r)/12
    rf_m = (1+rf_annual)**(1/12)-1
    vol_m = r.std(ddof=1)
    bvar = benchmark.var(ddof=1)
    beta = r.cov(benchmark)/bvar if bvar > 1e-15 else np.nan
    alpha_m = (r-rf_m).mean()-beta*(benchmark-rf_m).mean() if np.isfinite(beta) else np.nan
    active = r-benchmark
    te = active.std(ddof=1)*np.sqrt(12)
    sharpe = (r-rf_m).mean()/vol_m*np.sqrt(12) if vol_m > 1e-12 else np.nan
    treynor = (r.mean()-rf_m)*12/beta if np.isfinite(beta) and abs(beta)>1e-8 else np.nan
    mdd = (pd.concat([pd.Series([1.0], index=[r.index[0]-pd.Timedelta(days=1)]),growth]).pipe(lambda g: g/g.cummax()-1)).min()
    return {
        '누적 수익률':growth.iloc[-1]-1,
        'CAGR':growth.iloc[-1]**(1/years)-1,
        '벤치마크 CAGR':bgrowth.iloc[-1]**(1/years)-1,
        '벤치마크 대비 CAGR 차이':growth.iloc[-1]**(1/years)-bgrowth.iloc[-1]**(1/years),
        '연환산 변동성':vol_m*np.sqrt(12),
        'Sharpe':sharpe,
        'Beta':beta,
        'Jensen Alpha (연율 근사)':alpha_m*12 if np.isfinite(alpha_m) else np.nan,
        'Treynor (연율 근사)':treynor,
        'Information Ratio':active.mean()*12/te if te>1e-12 else np.nan,
        'Tracking Error (연율)':te,
        'MDD':mdd,
        'VaR 95% (월간, 역사적)':max(0.0,-float(r.quantile(.05))),
        'VaR 99% (월간, 역사적)':max(0.0,-float(r.quantile(.01))),
    }

def analyze_benchmarks(portfolio, benchmarks, rf_annual):
    """Returns benchmark-specific metrics on a common aligned monthly sample.

    benchmarks: DataFrame with columns e.g. 'KOSPI200', 'S&P500 (KRW)'.
    """
    frame = pd.concat([portfolio.rename('Portfolio'),benchmarks],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(frame)<6:
        raise ValueError('두 벤치마크를 포함한 공통 월간 데이터가 6개월 미만입니다.')
    return pd.DataFrame({name:analyze(frame['Portfolio'],frame[name],rf_annual)
                         for name in benchmarks.columns}).T

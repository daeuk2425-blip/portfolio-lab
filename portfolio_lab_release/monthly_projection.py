"""Historical monthly hypothetical wealth paths; no forecast or actual transaction history."""
import numpy as np
import pandas as pd

def monthly_wealth(portfolio_returns, initial_capital, benchmarks=None):
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('초기 투자금은 0보다 커야 합니다.')
    r = pd.Series(portfolio_returns, dtype=float).dropna()
    if r.empty or (r <= -1).any():
        raise ValueError('유효한 월간 수익률이 필요합니다.')
    out = pd.DataFrame(index=r.index)
    out['월간 수익률'] = r
    out['월말 평가금액'] = initial_capital * (1 + r).cumprod()
    out['월간 손익'] = out['월말 평가금액'].diff().fillna(out['월말 평가금액'].iloc[0] - initial_capital)
    out['누적 수익률'] = out['월말 평가금액'] / initial_capital - 1
    if benchmarks is not None:
        for name, b in benchmarks.items():
            aligned = pd.Series(b).reindex(r.index)
            if aligned.isna().any():
                raise ValueError(f'{name}의 분석기간 데이터가 부족합니다.')
            out[f'{name} 평가금액'] = initial_capital * (1 + aligned).cumprod()
            out[f'{name} 누적 수익률'] = out[f'{name} 평가금액'] / initial_capital - 1
    return out

def allocation_amounts(allocation, capital):
    result = allocation.copy()
    result['초기 배분금액(KRW)'] = result['비중(%)'].astype(float) / 100 * capital
    return result


def monthly_prices_for_dates(daily_prices, start_date, end_date):
    """Return start-date anchor and monthly/end-date prices, using available common trading dates."""
    p = daily_prices.copy().sort_index()
    p.index = pd.DatetimeIndex(p.index).tz_localize(None).normalize()
    p = p.loc[~p.index.duplicated(keep="last")]
    p = p.ffill().dropna(how="any")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    p = p.loc[(p.index >= start) & (p.index <= end)]
    if len(p) < 2:
        raise ValueError("선택한 기간에 공통 거래가격이 부족합니다. 날짜를 변경해 주세요.")
    anchor = p.iloc[[0]]
    monthly = p.groupby(p.index.to_period("M")).tail(1)
    sampled = pd.concat([anchor, monthly]).loc[lambda x: ~x.index.duplicated(keep="first")].sort_index()
    if len(sampled) < 2:
        raise ValueError("월별 수익률을 계산할 가격이 부족합니다.")
    return sampled, anchor.index[0].date(), sampled.index[-1].date()

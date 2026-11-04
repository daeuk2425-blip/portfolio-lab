"""Actual holdings valuation. Does not infer realized P&L or historical returns."""
import numpy as np
import pandas as pd

COLUMNS = ['종목명','티커','자산군','통화','보유수량','평균매입단가','현재가','현재가 기준일','환율(KRW/통화)']
CATEGORIES = ['국내주식','해외주식','국내ETF','해외ETF','암호화폐','채권','현금·예금','기타']

def calculate_holdings(frame):
    if frame is None or frame.empty:
        raise ValueError('보유 종목을 한 개 이상 입력하세요.')
    df=frame.copy()
    for c in COLUMNS:
        if c not in df: raise ValueError(f'필수 열 누락: {c}')
    for c in ['보유수량','평균매입단가','현재가','환율(KRW/통화)']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
        if df[c].isna().any() or (df[c]<0).any(): raise ValueError(f'{c}에는 0 이상의 숫자를 입력하세요.')
    if (df['보유수량']<=0).any() or (df['환율(KRW/통화)']<=0).any():
        raise ValueError('보유수량과 환율은 0보다 커야 합니다.')
    df['통화']=df['통화'].astype(str).str.upper().str.strip()
    if not df['통화'].isin(['KRW','USD']).all(): raise ValueError('현재는 KRW 및 USD 자산만 지원합니다.')
    if ((df['통화']=='KRW') & (~np.isclose(df['환율(KRW/통화)'],1))).any():
        raise ValueError('KRW 자산의 환율은 1로 입력하세요.')
    df['투자원금(KRW)']=df['보유수량']*df['평균매입단가']*df['환율(KRW/통화)']
    df['평가금액(KRW)']=df['보유수량']*df['현재가']*df['환율(KRW/통화)']
    df['평가손익(KRW)']=df['평가금액(KRW)']-df['투자원금(KRW)']
    df['평가수익률']=np.where(df['투자원금(KRW)']>0,df['평가손익(KRW)']/df['투자원금(KRW)'],np.nan)
    total=df['평가금액(KRW)'].sum()
    if total<=0: raise ValueError('평가금액 합계가 0보다 커야 합니다.')
    df['현재비중']=df['평가금액(KRW)']/total
    cost=df['투자원금(KRW)'].sum()
    summary={'total_value':float(total),'total_cost':float(cost),'unrealized_pnl':float(df['평가손익(KRW)'].sum()),'unrealized_return':float(df['평가손익(KRW)'].sum()/cost) if cost>0 else np.nan,'top_weight':float(df['현재비중'].max()),'hhi':float((df['현재비중']**2).sum()),'count':len(df)}
    return df,summary


def commentary(df,summary):
    top=df.loc[df['현재비중'].idxmax()]
    concentration=('상위 종목 비중이 40%를 넘으므로 개별 자산 변동에 대한 민감도가 높습니다.' if summary['top_weight']>.4 else '최대 종목 비중은 40% 이하이지만 자산 간 상관관계에 따라 실질 분산 수준이 달라질 수 있습니다.')
    foreign=float(df.loc[df['통화']=='USD','현재비중'].sum())
    fx=(f'USD 표시 자산 비중은 {foreign:.1%}입니다. 매입 당시 환율이 아닌 입력 환율로 원가를 환산하므로 표시 손익은 실제 원화 기준 투자손익과 다를 수 있습니다.' if foreign else '입력된 보유자산은 모두 원화 표시 자산입니다.')
    return [f"현재 평가금액은 {summary['total_value']:,.0f}원이며, 입력 원가 대비 미실현 평가손익은 {summary['unrealized_pnl']:+,.0f}원입니다. 거래세·수수료·분배금·실현손익은 제외했습니다.",f"가장 큰 보유자산은 {top['종목명']}({top['현재비중']:.1%})입니다. {concentration}",fx,'현재 보유잔고의 평가수익률은 기간별 투자수익률(TWR/CAGR)이 아닙니다. 매수·매도·입출금 거래내역 없이 Sharpe, MDD, VaR를 실제 계좌 성과로 계산하지 않습니다.']

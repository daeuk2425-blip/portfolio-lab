"""Deterministic, evidence-aware Korean research paragraphs; no LLM or invented citations."""
import numpy as np

def pct(x):
    return f'{x*100:.1f}%' if np.isfinite(x) else '산출 불가'

def write_analysis(allocation, stats, comparison, observations, macro_view, months):
    ranked=allocation.sort_values('비중(%)',ascending=False)
    top=ranked.iloc[0]
    concentration=float((allocation['비중(%)']/100).pow(2).sum())
    top2=float(ranked['비중(%)'].head(2).sum())
    cagr=stats['CAGR']; diff=stats['벤치마크 대비 CAGR 차이']; mdd=stats['MDD']
    outperform='상회' if diff>=0 else '하회'
    thesis=(f"본 포트폴리오는 {len(allocation)}개 자산으로 구성되며, 최대 편입 자산은 {top['종목명']}({top['비중(%)']:.1f}%)이다. "
        f"상위 두 자산의 비중은 {top2:.1f}%이고, 비중 제곱합(HHI)은 {concentration:.3f}으로 나타난다. "
        f"이는 자산 수 자체보다 실제 위험 노출이 일부 자산에 집중될 가능성을 점검해야 한다는 의미다. "
        f"관측된 {months}개월 동안 연환산 수익률은 {pct(cagr)}로, 원화 환산 SPY 벤치마크를 {abs(diff)*100:.1f}%p {outperform}했다. "
        "다만 과거 상대성과가 미래의 우위를 보장하지는 않으며, 비교 대상의 통화·거래비용·분배금 처리 차이를 함께 고려해야 한다.")
    vol=stats['연환산 변동성']; sharpe=stats['Sharpe']; var95=stats['VaR 95% (월간, 역사적)']
    risk=(f"연환산 변동성은 {pct(vol)}, 최대낙폭(MDD)은 {pct(mdd)}, Sharpe 비율은 {sharpe:.2f}이다. "
        f"과거 월간 수익률 하위 5% 분위수에 근거한 95% VaR는 {pct(var95)}로 추정된다. "
        "VaR는 지정된 확률 수준의 손실 임계값이지 가능한 최대 손실이 아니다. "
        "특히 급격한 시장 충격과 상관관계 상승이 동시에 나타나는 경우 과거 표본 기반 위험 추정은 실제 손실을 과소평가할 수 있다. ")
    if len(comparison)>1:
        best=comparison['CAGR'].idxmax(); least=comparison['MDD'].idxmax()
        risk+=(f"비교한 리밸런싱 방식 중 과거 CAGR이 가장 높았던 방식은 '{best}', 최대낙폭이 가장 작았던 방식은 '{least}'이다. "
               "두 결과가 같지 않을 수 있으므로 수익률 하나만으로 리밸런싱 주기를 결정하기 어렵다. "
               "본 비교에는 설정한 거래비용만 반영되며 세금·슬리피지는 제외된다.")
    tickers=allocation['종목 코드'].astype(str).tolist()
    foreign=float(allocation.loc[~allocation['종목 코드'].str.endswith(('.KS','.KQ')) & ~allocation['종목 코드'].isin(['DEPOSIT','CMA']),'비중(%)'].sum())
    crypto=float(allocation.loc[allocation['종목 코드'].isin(['BTC-USD','ETH-USD']),'비중(%)'].sum())
    domestic=float(allocation.loc[allocation['종목 코드'].str.endswith(('.KS','.KQ')),'비중(%)'].sum())
    sector=(f"포트폴리오 내 국내 상장주식 비중은 {domestic:.1f}%, 달러 표시 시장성 자산 비중은 {foreign:.1f}%, "
        f"암호화폐 비중은 {crypto:.1f}%이다. "
        "달러 표시 자산의 원화 성과는 현지 자산가격뿐 아니라 원/달러 환율 변화의 영향을 받는다. "
        "암호화폐는 위험회피 국면에서 주식과 함께 하락할 수 있어 항상 분산 효과를 제공한다고 가정할 수 없다. "
        "종목별 영업실적이나 증권사 목표주가에 관한 주장은 별도의 실제 공시·리서치 검증 없이 제시하지 않는다.")
    if observations:
        names=', '.join(str(o.get('name','')) for o in observations[:3])
        macro=(f"조회된 공식 통계 관측치는 {names} 등이다. "+(macro_view or '')[:600])
    else:
        macro=("이번 실행에서는 검증된 ECOS·KOSIS 최신 관측치를 확보하지 못했다. 따라서 현재 물가·금리·경기의 방향을 확정적으로 진술하지 않는다. "
               "금리 변화는 할인율뿐 아니라 기업 이익과 경기 기대를 통해서도 가격에 영향을 미친다. "+(macro_view or '')[:450])
    conclusion=("향후 점검의 우선순위는 ① 상위 편입 자산의 집중 위험 ② 달러 노출의 환율 효과 ③ 급락 국면의 동조화 ④ 거래비용을 포함한 리밸런싱 효과다. "
                "이 보고서는 과거 가격과 입력 가정에 대한 조건부 분석이며, 투자 판단을 대체하지 않는다.")
    return {'thesis':thesis,'risk_interpretation':risk,'company_sector':sector,'macro_scenarios':macro,'conclusion':conclusion}

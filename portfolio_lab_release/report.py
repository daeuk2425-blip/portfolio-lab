"""Portfolio Lab research-style, dynamically paginated PDF report.
Compatible with app_v2.py's make_pdf call. Backtest only, not holdings P&L.
"""
import io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from xml.sax.saxutils import escape

NAVY=colors.HexColor('#163755'); BLUE=colors.HexColor('#2876A5'); PALE=colors.HexColor('#EAF2F8'); GRAY=colors.HexColor('#607080')
FONT_PATHS=['/usr/share/fonts/truetype/nanum/NanumGothic.ttf','/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf','/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']

def get_font():
    for p in FONT_PATHS:
        if os.path.isfile(p):
            try:
                if 'PLKorean' not in pdfmetrics.getRegisteredFontNames(): pdfmetrics.registerFont(TTFont('PLKorean',p))
                font_manager.fontManager.addfont(p)
                plt.rcParams['font.family']=font_manager.FontProperties(fname=p).get_name()
                plt.rcParams['axes.unicode_minus']=False
                return 'PLKorean'
            except Exception: continue
    raise RuntimeError('한글 글꼴을 찾을 수 없습니다. packages.txt에 fonts-nanum을 추가하세요.')

def make_pdf(allocation, selected_label, stats, strategy_table, monthly, portfolio, benchmark, corr, sources, assumptions, analyst=None, macro_view=None, observations=None, macro_chart=None, macro_label='', macro_transform='원자료', benchmark_kospi200=None, benchmark_stats=None, monthly_result=None, initial_capital=None, capital_allocation=None):
    font=get_font(); buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=(21*cm,29.7*cm),leftMargin=1.8*cm,rightMargin=1.8*cm,topMargin=2*cm,bottomMargin=1.8*cm)
    s=getSampleStyleSheet()
    s.add(ParagraphStyle(name='PLTitle',fontName=font,fontSize=19,leading=27,textColor=NAVY,spaceAfter=10))
    s.add(ParagraphStyle(name='PLSection',fontName=font,fontSize=12,leading=18,textColor=NAVY,spaceBefore=15,spaceAfter=7,keepWithNext=True))
    s.add(ParagraphStyle(name='PLBody',fontName=font,fontSize=9,leading=15,spaceAfter=8))
    s.add(ParagraphStyle(name='PLSmall',fontName=font,fontSize=7.5,leading=11,spaceAfter=5))
    s.add(ParagraphStyle(name='PLCell',fontName=font,fontSize=7.5,leading=11))
    def P(x,style='PLBody'):return Paragraph(escape(str(x)),s[style])
    def section(title):story.append(P(title,'PLSection'))
    def fmt(x,pct=False):
        try:
            x=float(x)
            return ('{:.2%}' if pct else '{:.3f}').format(x) if np.isfinite(x) else '산출 불가'
        except (ValueError,TypeError):return '산출 불가'
    def table(rows,widths=None):
        t=Table([[P(v,'PLCell') for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('LINEBELOW',(0,0),(-1,0),.9,BLUE),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#D9E3EA')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),5)]))
        return t
    def chart(df,title,ylabel='초기 1 기준',height=6):
        df=df.replace([np.inf,-np.inf],np.nan).dropna(how='all')
        fig,ax=plt.subplots(figsize=(9,3.7))
        for col in df.columns:ax.plot(df.index,df[col],label=str(col),linewidth=1.8)
        ax.set_title(title,fontsize=12,loc='left',fontweight='bold')
        ax.set_ylabel(ylabel,fontsize=9)
        ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8)
        fig.autofmt_xdate();fig.tight_layout()
        out=io.BytesIO();fig.savefig(out,format='png',dpi=145);plt.close(fig);out.seek(0)
        return Image(out,width=17.2*cm,height=height*cm)
    story=[]
    story.append(P('PORTFOLIO LAB  |  투자 포트폴리오 리서치','PLTitle'))
    story.append(P(f'분석기간 {monthly.index.min():%Y-%m} ~ {monthly.index.max():%Y-%m}  |  리밸런싱: {selected_label}  |  기준통화: KRW','PLSmall'))
    story.append(P('본 자료는 입력 비중을 기반으로 과거 가격을 재현한 가상 백테스트입니다. 실제 보유 수량·매입단가·평가손익을 추정하거나 표시하지 않습니다.'))
    section('01  포트폴리오 구성 및 핵심 성과')
    rows=[['편입 자산','티커','목표 비중']]
    for _,r in allocation.iterrows():rows.append([str(r['종목명']),str(r['종목 코드']),f"{float(r['비중(%)']):.1f}%"])
    story.append(table(rows,[7*cm,6.1*cm,4.1*cm]))
    story.append(Spacer(1,9))
    keys=['누적 수익률','CAGR','연환산 변동성','Sharpe','MDD','VaR 95% (월간, 역사적)']
    percent={'누적 수익률','CAGR','연환산 변동성','MDD','VaR 95% (월간, 역사적)'}
    story.append(table([['핵심 지표','값']]+[[k,fmt(stats.get(k),k in percent)] for k in keys],[10.5*cm,6.7*cm]))
    if analyst and analyst.get('thesis'):
        section('핵심 투자 논지');story.append(P(analyst['thesis']))
    section('02  KOSPI200 및 S&P500 성과 비교')
    common=pd.concat([portfolio.rename('포트폴리오'),benchmark.rename('S&P500 (SPY ETF, KRW)')],axis=1)
    if benchmark_kospi200 is not None:common['KOSPI200 (KODEX 200 ETF 대용)']=benchmark_kospi200
    common=common.dropna()
    if len(common)>1:
        growth=(1+common).cumprod();story.append(chart(growth,'누적 성장률 비교 (월말 기준)'))
        years=len(common)/12
        rows=[['비교대상','누적 수익률','연환산 수익률']]
        for col in growth.columns:
            total=growth[col].iloc[-1]-1;cagr=growth[col].iloc[-1]**(1/years)-1
            rows.append([col,fmt(total,True),fmt(cagr,True)])
        story.append(table(rows,[9*cm,4.1*cm,4.1*cm]))
    story.append(P('두 지수는 직접적인 총수익률 지수가 아닌 ETF 조정가격을 사용한 대용 비교입니다. SPY는 원화 환산, KODEX 200은 원화 가격을 사용합니다. ETF 보수·추적오차·분배금 처리 차이로 실제 지수와 차이가 날 수 있습니다.','PLSmall'))
    section('03  위험조정성과 및 벤치마크 민감도')
    risk=['Sharpe','Beta','Treynor (연율 근사)','Jensen Alpha (연율 근사)','Information Ratio','Tracking Error (연율)','MDD','VaR 95% (월간, 역사적)','VaR 99% (월간, 역사적)']
    pct={'Treynor (연율 근사)','Jensen Alpha (연율 근사)','Tracking Error (연율)','MDD','VaR 95% (월간, 역사적)','VaR 99% (월간, 역사적)'}
    story.append(table([['위험 및 성과 지표','값']]+[[k,fmt(stats.get(k),k in pct)] for k in risk],[10.5*cm,6.7*cm]))
    if benchmark_stats is not None:
        frame=pd.DataFrame(benchmark_stats)
        if not frame.empty:
            rows=[['벤치마크','Beta','Treynor','Jensen Alpha','정보비율']]
            for name,r in frame.iterrows():rows.append([str(name),fmt(r.get('Beta')),fmt(r.get('Treynor (연율 근사)'),True),fmt(r.get('Jensen Alpha (연율 근사)'),True),fmt(r.get('Information Ratio'))])
            story.append(Spacer(1,9));story.append(table(rows,[5.8*cm,2.4*cm,3*cm,3.2*cm,2.8*cm]))
    if len(portfolio)>1:
        wealth=(1+portfolio).cumprod();wealth=pd.concat([pd.Series([1.0],index=[portfolio.index[0]-pd.Timedelta(days=1)]),wealth]);dd=wealth/wealth.cummax()-1
        story.append(Spacer(1,12));story.append(chart(dd.to_frame('포트폴리오 낙폭'),'최대낙폭(MDD) 경로',ylabel='낙폭',height=5.3))
    story.append(P('Sharpe·Treynor·Jensen Alpha는 가정 무위험수익률을 사용합니다. VaR는 월간 과거 수익률 분위수 기반이며 최대 손실을 뜻하지 않습니다. 월간 표본의 99% VaR는 특히 불안정합니다.','PLSmall'))
    if analyst and analyst.get('risk_interpretation'):
        section('위험 해석');story.append(P(analyst['risk_interpretation']))
    section('04  리밸런싱 전략 및 자산 간 상관관계')
    if strategy_table is not None and not strategy_table.empty:
        rows=[['리밸런싱','CAGR','MDD','Sharpe']]
        for name,r in strategy_table.iterrows():rows.append([str(name),fmt(r.get('CAGR'),True),fmt(r.get('MDD'),True),fmt(r.get('Sharpe'))])
        story.append(table(rows,[5.7*cm,3.8*cm,3.8*cm,3.9*cm]))
    if corr is not None and not corr.empty:
        names=list(corr.columns)
        # Chunk matrix horizontally to keep each table readable, without dropping holdings.
        for start in range(0,len(names),5):
            subset=names[start:start+5]
            rows=[['종목']+[str(x)[:12] for x in subset]]
            for i in names:rows.append([str(i)[:12]]+[fmt(corr.loc[i,j]) for j in subset])
            story.append(Spacer(1,10));story.append(table(rows,[4.2*cm]+[13*cm/len(subset)]*len(subset)))
    story.append(P('예금·CMA의 수익률을 고정 가정한 경우 상관계수가 산출되지 않을 수 있습니다.','PLSmall'))
    if analyst and analyst.get('company_sector'):
        section('자산 및 업종 노출 해석');story.append(P(analyst['company_sector']))
    section('05  종합 평가 및 분석 가정')
    if analyst and analyst.get('conclusion'):story.append(P(analyst['conclusion']))
    if analyst and analyst.get('macro_scenarios'):
        section('조건부 거시경제 시나리오');story.append(P(analyst['macro_scenarios']))
    section('자료 및 방법론')
    for item in assumptions or []:story.append(P('• '+str(item),'PLSmall'))
    story.append(P('가격: Yahoo Finance/yfinance를 통해 조회한 조정가격. 국내 상장 종목 목록: FinanceDataReader 등. 외부 제공자의 데이터 오류·결측·수정 가능성이 있습니다.','PLSmall'))
    story.append(P('실제 투자손익, 보유 수량, 취득단가, 세금, 배당 재투자 실현 가능성, 슬리피지, 개별 종목의 공시 검토 및 미래 전망은 본 백테스트의 검증 범위 밖입니다. 분석 문장은 규칙 기반이며 독립적인 기업 리서치가 아닙니다.','PLSmall'))
    for title,url in (sources or [])[:10]:story.append(P(str(title)+' | '+str(url),'PLSmall'))
    story.append(P('본 보고서는 정보 제공용 과거 시뮬레이션이며 투자 권유가 아닙니다.','PLSmall'))
    if monthly_result is not None and not monthly_result.empty:
        story.append(PageBreak())
        section('06  월별 투자성과 및 초기 투자금 배분')
        story.append(P(f'초기 투자금액: {initial_capital:,.0f}원 | 분석 월수: {len(monthly_result)}개월', 'PLSmall'))
        if capital_allocation is not None and not capital_allocation.empty:
            rows=[['자산', '목표 비중', '초기 배분금액']]
            for _, a in capital_allocation.iterrows():
                rows.append([str(a['종목명']),f"{a['비중(%)']:.1f}%",f"{a['초기 배분금액(KRW)']:,.0f}원"])
            story.append(table(rows,[6.8*cm,4*cm,6.4*cm]))
        path=monthly_result[['월말 평가금액','S&P500 (SPY, KRW) 평가금액','KOSPI200 (KODEX 200 ETF) 평가금액']].copy()
        story.append(Spacer(1,12))
        story.append(chart(path,'월별 누적 평가금액 비교',ylabel='원',height=5.5))
        rows=[['월','월간 수익률','월간 손익','월말 평가금액','누적 수익률']]
        for date,r in monthly_result.iterrows():
            rows.append([f'{date:%Y-%m}',f"{r['월간 수익률']:+.2%}",f"{r['월간 손익']:+,.0f}",f"{r['월말 평가금액']:,.0f}",f"{r['누적 수익률']:+.2%}"])
        story.append(Spacer(1,10))
        story.append(table(rows,[3*cm,3.3*cm,3.6*cm,4.2*cm,3.1*cm]))
        best=monthly_result['월간 수익률'].idxmax();worst=monthly_result['월간 수익률'].idxmin()
        story.append(P(f"최고 성과 월: {best:%Y-%m} ({monthly_result.loc[best,'월간 수익률']:+.2%}) | 최저 성과 월: {worst:%Y-%m} ({monthly_result.loc[worst,'월간 수익률']:+.2%})",'PLSmall'))
        story.append(P('매월 말 가상 평가금액은 초기 투자금에 월별 시뮬레이션 수익률을 복리 적용했습니다. 실제 매수 수량, 매입 시점, 입출금 및 실현손익과 다릅니다. 종목별 초기 배분금액은 목표 비중 기준이며 실제 체결금액이 아닙니다.','PLSmall'))
    def footer(c,d):
        c.setStrokeColor(colors.HexColor('#C8D7E4'));c.line(1.8*cm,1.42*cm,19.2*cm,1.42*cm)
        c.setFont(font,8);c.setFillColor(GRAY);c.drawString(1.8*cm,1.06*cm,'PORTFOLIO LAB  |  Research & Risk Analytics');c.drawRightString(19.2*cm,1.06*cm,str(d.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return buf.getvalue()

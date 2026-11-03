import io, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _fontpath in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf']:
    if os.path.exists(_fontpath):
        font_manager.fontManager.addfont(_fontpath)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=_fontpath).get_name()
        break
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import simpleSplit
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from xml.sax.saxutils import escape

FONT_CANDIDATES = ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']

def get_font():
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                pdfmetrics.registerFont(TTFont('ReportKorean', path))
                return 'ReportKorean'
            except Exception: pass
    raise RuntimeError('한글 PDF 글꼴이 없습니다. Colab에서 !apt-get -qq install fonts-nanum 을 실행해 주세요.')

def plot_image(df, title, width=16, height=6):
    fig, ax = plt.subplots(figsize=(8, 3.1))
    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax.set_title(title)
    fig.tight_layout()
    b=io.BytesIO(); fig.savefig(b, format='png', dpi=125); plt.close(fig); b.seek(0)
    return Image(b, width=width*cm, height=height*cm)

def make_pdf(allocation, selected_label, stats, strategy_table, monthly, portfolio, benchmark, corr, sources, assumptions, analyst=None, macro_view=None, observations=None, macro_chart=None, macro_label="", macro_transform="원자료"):
    font = get_font()
    b=io.BytesIO()
    doc=SimpleDocTemplate(b, pagesize=(21*cm,29.7*cm), rightMargin=2*cm,leftMargin=2*cm,topMargin=1.7*cm,bottomMargin=1.7*cm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='KTitle', fontName=font, fontSize=18, leading=24, spaceAfter=12, textColor=colors.HexColor('#15304C')))
    styles.add(ParagraphStyle(name='KHead', fontName=font, fontSize=12, leading=18, spaceBefore=12, spaceAfter=8, textColor=colors.HexColor('#15304C')))
    styles.add(ParagraphStyle(name='KBody', fontName=font, fontSize=9, leading=15, spaceAfter=8))
    styles.add(ParagraphStyle(name='KSmall', fontName=font, fontSize=8, leading=12, spaceAfter=5))
    styles.add(ParagraphStyle(name='KResearch', fontName=font, fontSize=8.4, leading=13.6, spaceAfter=7))
    def P(x, style='KBody'): return Paragraph(escape(str(x)), styles[style])
    def tab(rows, widths=None):
        t=Table([[P(v,'KSmall') for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF5')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#D5DDE5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)]))
        return t
    def fmt(v, pct=False):
        return '산출 불가' if not np.isfinite(v) else (f'{v:.2%}' if pct else f'{v:.3f}')
    story=[]
    # page 1
    story += [P('PORTFOLIO LAB | 개인 투자 분석 보고서','KTitle'), P('01  포트폴리오 개요','KHead'), P(f'분석 기간: {monthly.index.min():%Y-%m} ~ {monthly.index.max():%Y-%m}  |  전략: {selected_label}'), P('사용자가 선택한 자산 비중과 과거 월말 가격을 이용한 가상 백테스트입니다.')]
    story.append(tab([['종목명','티커','목표 비중']]+[[str(r['종목명']),str(r['종목 코드']),f"{r['비중(%)']:.1f}%"] for _,r in allocation.iterrows()],[6*cm,6*cm,5*cm]))
    story += [P('핵심 성과','KHead'),tab([['지표','포트폴리오'],['연환산 수익률(CAGR)',fmt(stats['CAGR'],True)],['벤치마크 CAGR',fmt(stats['벤치마크 CAGR'],True)],['벤치마크 대비 CAGR 차이',fmt(stats['벤치마크 대비 CAGR 차이'],True)],['최대낙폭(MDD)',fmt(stats['MDD'],True)]],[9*cm,8*cm])]
    story += [P('기준 벤치마크: SPY ETF의 원화 환산 총수익률 대용 가격. ETF 가격과 지수 성과는 다를 수 있습니다.','KSmall')]
    if analyst:
        story += [P('포트폴리오 리서치 | 투자 논지','KHead'), P(analyst['thesis'],'KResearch')]
    story.append(PageBreak())
    # page 2
    story += [P('02  성과 및 위험 분석','KTitle')]
    risk_names=['연환산 변동성','Sharpe','Beta','Jensen Alpha (연율 근사)','Treynor (연율 근사)','VaR 95% (월간, 역사적)','VaR 99% (월간, 역사적)','MDD']
    pct_names={'연환산 변동성','Jensen Alpha (연율 근사)','Treynor (연율 근사)','VaR 95% (월간, 역사적)','VaR 99% (월간, 역사적)','MDD'}
    story.append(tab([['지표','값']]+[[n,fmt(stats[n],n in pct_names)] for n in risk_names],[10*cm,7*cm]))
    story += [P('리밸런싱 전략 비교','KHead'),tab([['전략','CAGR','MDD','Sharpe']]+[[str(i),fmt(r['CAGR'],True),fmt(r['MDD'],True),fmt(r['Sharpe'])] for i,r in strategy_table.iterrows()],[5*cm,4*cm,4*cm,4*cm])]
    story += [P('VaR는 과거 월간 수익률의 하위 분위수로 추정한 손실 임계값입니다. 최대 손실을 뜻하지 않으며 특히 표본이 짧으면 99% VaR가 불안정합니다. Treynor는 베타가 0에 가까우면 산출하지 않습니다.','KSmall')]
    if analyst:
        story += [P('포트폴리오 리서치 | 위험 해석','KHead'),P(analyst['risk_interpretation'],'KResearch')]
    story.append(PageBreak())
    # page 3
    story += [P('03  성과 경로 및 기업·산업 분석','KTitle')]
    growth=pd.DataFrame({'Portfolio':(1+portfolio).cumprod(),'Benchmark':(1+benchmark).cumprod()})
    story.append(plot_image(growth,'Cumulative growth',16,5.3))
    story += [P('자산별 월간 수익률 상관계수','KHead')]
    names=list(corr.columns)[:5]
    rows=[['자산']+[str(x)[:10] for x in names]]
    for i in names:
        rows.append([str(i)[:10]]+[('—' if pd.isna(corr.loc[i,j]) else f'{corr.loc[i,j]:.2f}') for j in names])
    widths=[17*cm/(len(names)+1)]*(len(names)+1)
    story.append(tab(rows,widths))
    story.append(P('고정 금리 가정 예금·CMA의 상관계수는 정의되지 않을 수 있습니다.','KSmall'))
    if analyst:
        story += [P('포트폴리오 리서치 | 자산 노출 분석','KHead'), P(analyst['company_sector'],'KResearch')]
    else:
        story += [P('기업·산업 리서치 미실행','KHead'), P('실제 공시·뉴스 검색을 사용한 AI 분석은 API 키를 입력해야 생성됩니다.','KSmall')]
    story.append(PageBreak())
    # page 4
    story += [P('04  거시경제 통계 및 투자 시사점','KTitle')]
    if macro_chart is not None and len(macro_chart.dropna()) >= 2:
        frame = macro_chart.dropna().tail(36).to_frame(name=str(macro_label)[:20])
        story.append(plot_image(frame, 'Official macro series - ' + str(macro_transform), 16, 5.0))
        story.append(P(f'출처: 공식 {"ECOS/KOSIS"} 조회 원자료 | 지표: {macro_label} | 가공: {macro_transform} | 마지막 관측: {frame.index[-1]:%Y-%m}', 'KSmall'))
    else:
        story.append(P('원자료 시계열이 조회되지 않아 실제 거시경제 차트를 생략했습니다.', 'KSmall'))
    if analyst:
        story += [P('거시경제 위험 시나리오','KHead'),P(analyst['macro_scenarios'],'KResearch'),
                  P('종합 평가','KHead'),P(analyst['conclusion'],'KResearch')]
        story.append(P('작성 방식: 공식 가격 데이터·사용자 가정·규칙 기반 정량 해석 | 생성형 AI 미사용', 'KSmall'))
    else:
        story += [P('일반적인 위험 점검 (규칙 기반)','KHead')]
        codes=set(allocation['종목 코드'].astype(str))
        if any(x.endswith(('.KS','.KQ')) for x in codes): story.append(P('국내 주식: 국내 경기 및 업종 실적 민감도를 점검하세요.','KSmall'))
        if any(x not in ('DEPOSIT','CMA') and not x.endswith(('.KS','.KQ')) for x in codes): story.append(P('달러 표시 자산: 환율 변동이 원화 수익률에 영향을 미칩니다.','KSmall'))
        if any(x in codes for x in ('BTC-USD','ETH-USD')): story.append(P('암호화폐: 유동성, 규제 및 급격한 변동성 위험을 고려하세요.','KSmall'))
    if macro_view:
        story += [P('공공통계 기반 거시경제 해석 (규칙 기반)','KHead'), P(macro_view[:800], 'KResearch')]
    story += [P('검증 가능한 원문 출처','KHead')]
    if sources:
        for title,url in sources[:5]:
            # Keep a working clickable URL, with a shorter visible label.
            safe_url=escape(str(url), {'"':'&quot;'})
            label=escape(str(title)[:105])
            story.append(Paragraph(f'<link href="{safe_url}" color="#1A5689">{label}</link>',styles['KSmall']))
    else:
        story.append(P('검증 가능한 검색 출처 없음. 실제 자료를 확인했다고 주장하지 않습니다.','KSmall'))
    story += [P('분석 가정과 한계','KHead')]
    for a in assumptions[:3]: story.append(P('• '+a,'KSmall'))
    story.append(P('본 보고서는 정보 제공 목적의 과거 시뮬레이션이며 투자 권유가 아닙니다. 경제지표는 발표 시점과 수정 가능성을 확인해야 합니다. 규칙 기반 해석은 독립적인 애널리스트 조사와 다릅니다.','KSmall'))
    def footer(canvas, document):
        canvas.setFont(font,8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2*cm,1.1*cm,'PORTFOLIO LAB  |  Historical simulation')
        canvas.drawRightString(19*cm,1.1*cm,f'{document.page} / 4')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return b.getvalue()

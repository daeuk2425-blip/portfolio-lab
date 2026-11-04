"""IBK-inspired research layout for actual holdings, dynamically paginated."""
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
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from xml.sax.saxutils import escape
from holdings import commentary

NAVY=colors.HexColor('#183C5A'); BLUE=colors.HexColor('#2477A6'); PALE=colors.HexColor('#E9F2F8')
FONT_PATHS=['/usr/share/fonts/truetype/nanum/NanumGothic.ttf','/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']

def font_setup():
    for path in FONT_PATHS:
        if os.path.isfile(path):
            try:
                if 'HoldingKorean' not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont('HoldingKorean',path))
                font_manager.fontManager.addfont(path)
                plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name()
                plt.rcParams['axes.unicode_minus']=False
                return 'HoldingKorean'
            except Exception: pass
    raise RuntimeError('한글 글꼴이 없습니다. packages.txt에 fonts-nanum을 추가하세요.')

def make_holdings_pdf(df, summary, valuation_date, price_source='사용자 입력'):
    font=font_setup();buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=(21*cm,29.7*cm),leftMargin=1.7*cm,rightMargin=1.7*cm,topMargin=2*cm,bottomMargin=1.8*cm)
    ss=getSampleStyleSheet()
    for name,size,leading,color,space in [('H1',19,26,NAVY,12),('H2',12,18,NAVY,9),('BodyK',9,15,colors.black,7),('SmallK',7.5,12,colors.HexColor('#4C6274'),5),('CellK',7.2,11,colors.black,0)]:
        ss.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=space,wordWrap='CJK'))
    def p(s,sty='BodyK'):return Paragraph(escape(str(s)),ss[sty])
    def section(t):story.append(Spacer(1,10));story.append(p(t,'H2'))
    def table(rows,widths):
        t=Table([[p(v,'CellK') for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('LINEBELOW',(0,0),(-1,0),1,BLUE),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#D9E4EC')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        return t
    story=[p('PORTFOLIO LAB  |  보유자산 투자 리서치','H1'),p(f'평가기준일: {valuation_date}  |  기준통화: KRW  |  현재가 출처: {price_source}','SmallK'),p('실제 보유잔고의 미실현 평가손익 분석입니다. 거래내역을 사용한 과거 운용성과나 미래 전망이 아닙니다.','SmallK')]
    section('01  Executive Summary | 핵심 평가')
    story.append(table([['지표','값'],['총 평가금액',f"{summary['total_value']:,.0f} 원"],['입력 원가 합계',f"{summary['total_cost']:,.0f} 원"],['미실현 평가손익',f"{summary['unrealized_pnl']:+,.0f} 원"],['원가 대비 평가수익률',f"{summary['unrealized_return']:.2%}" if np.isfinite(summary['unrealized_return']) else '산출 불가'],['최대 종목 비중',f"{summary['top_weight']:.1%}"],['보유 종목 수',str(summary['count'])]],[9*cm,8.6*cm]))
    section('02  Holdings | 보유 종목별 평가')
    rows=[['종목 / 티커','수량','매입단가','현재가','평가금액(원)','손익(원)','비중']]
    for _,r in df.iterrows():
        rows.append([f"{r['종목명']} ({r['티커']})",f"{r['보유수량']:,.4g}",f"{r['평균매입단가']:,.2f}",f"{r['현재가']:,.2f}",f"{r['평가금액(KRW)']:,.0f}",f"{r['평가손익(KRW)']:+,.0f}",f"{r['현재비중']:.1%}"])
    story.append(table(rows,[4.1*cm,1.7*cm,2.2*cm,2.2*cm,2.9*cm,2.7*cm,1.8*cm]))
    story.append(p('매입단가·현재가는 각 종목의 표시통화 기준입니다. 평가금액과 손익은 입력한 동일 환율을 적용한 원화 값입니다.','SmallK'))
    section('03  Asset Allocation | 자산배분')
    groups=df.groupby('자산군',dropna=False)['평가금액(KRW)'].sum().sort_values(ascending=False)
    story.append(table([['자산군','평가금액(원)','비중']]+[[str(k),f'{v:,.0f}',f"{v/summary['total_value']:.1%}"] for k,v in groups.items()],[7*cm,6*cm,4.6*cm]))
    fig,ax=plt.subplots(figsize=(8.5,3.2))
    order=df.sort_values('평가금액(KRW)',ascending=True).tail(12)
    ax.barh(order['종목명'].astype(str),order['현재비중']*100,color='#2477A6')
    ax.set_xlabel('포트폴리오 비중 (%)');ax.grid(axis='x',alpha=.2);fig.tight_layout()
    plot=io.BytesIO();fig.savefig(plot,format='png',dpi=145);plt.close(fig);plot.seek(0)
    story.append(Image(plot,width=17*cm,height=6.4*cm))
    section('04  Analyst Commentary | 포트폴리오 진단')
    for paragraph in commentary(df,summary):story.append(p(paragraph))
    section('05  Methodology | 자료 및 분석 한계')
    for line in ['평가금액 = 보유수량 × 현재가 × 입력 환율','원가 = 보유수량 × 평균매입단가 × 입력 환율','평가손익 = 평가금액 - 원가; 비중 = 종목 평가금액 / 전체 평가금액','입력 환율은 현재가와 원가에 동일하게 적용됩니다. 해외자산의 정확한 원화 투자손익은 매수시점 환율·거래내역이 추가로 필요합니다.','세금, 수수료, 배당·분배금, 실현손익, 예금 이자, 미수금 및 외부 현금흐름은 반영하지 않습니다.','가격의 정확성과 기준일은 사용자가 확인해야 합니다. 실시간 가격을 보장하지 않습니다.']:
        story.append(p('• '+line,'SmallK'))
    story.append(p('본 보고서는 정보 제공용이며 투자 권유 또는 독립적인 기업 리서치가 아닙니다.','SmallK'))
    def footer(canvas,document):
        canvas.setFont(font,8);canvas.setFillColor(colors.HexColor('#718293'))
        canvas.drawString(1.7*cm,1.05*cm,'PORTFOLIO LAB  |  Holdings Research')
        canvas.drawRightString(19.3*cm,1.05*cm,str(document.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return buf.getvalue()

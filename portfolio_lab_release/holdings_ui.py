"""Streamlit interface for current holdings, separate from historical backtest."""
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from holdings import COLUMNS, CATEGORIES, calculate_holdings, commentary
from holdings_report import make_holdings_pdf

@st.cache_data(ttl=1800,show_spinner=False)
def latest_quote(ticker):
    t=yf.Ticker(ticker)
    history=t.history(period='10d',auto_adjust=False)
    if history.empty or 'Close' not in history:raise ValueError('가격 이력이 없습니다')
    currency=t.fast_info.get('currency')
    if currency not in ('KRW','USD'):
        raise ValueError(f'지원하지 않는 표시통화: {currency}. 가격을 직접 입력하세요.')
    date=history.index[-1].date().isoformat()
    return float(history['Close'].dropna().iloc[-1]),str(currency),date

@st.cache_data(ttl=1800,show_spinner=False)
def latest_fx():
    hist=yf.Ticker('KRW=X').history(period='10d')
    if hist.empty:raise ValueError('USD/KRW 환율을 조회하지 못했습니다.')
    return float(hist['Close'].dropna().iloc[-1]),hist.index[-1].date().isoformat()

def render_holdings():
    st.subheader('실제 보유자산 평가')
    st.caption('종목별 수량·평균매입단가·현재가를 입력해 평가금액과 미실현 손익을 계산합니다. 백테스트와는 별도의 분석입니다.')
    st.info('현재가와 환율은 자동 조회 후에도 수정할 수 있습니다. USD 원가에 현재 환율을 적용하므로 실제 원화 투자손익과 차이가 납니다. 실제 손익은 거래별 매입환율과 수수료가 필요합니다.')
    if 'holdings_data_v1' not in st.session_state:
        st.session_state.holdings_data_v1=pd.DataFrame([{'종목명':'삼성전자','티커':'005930.KS','자산군':'국내주식','통화':'KRW','보유수량':10.0,'평균매입단가':0.0,'현재가':0.0,'현재가 기준일':'','환율(KRW/통화)':1.0}])
    st.caption('예시 종목은 입력 방법을 보여주기 위한 것이며, 수량과 가격을 본인의 값으로 바꾸세요.')
    frame=st.data_editor(st.session_state.holdings_data_v1,column_order=COLUMNS,num_rows='dynamic',hide_index=True,use_container_width=True,key='holdings_editor_v1',column_config={
        '자산군':st.column_config.SelectboxColumn('자산군',options=CATEGORIES,required=True),
        '통화':st.column_config.SelectboxColumn('통화',options=['KRW','USD'],required=True),
        '보유수량':st.column_config.NumberColumn('보유수량',min_value=0.0,format='%.4f'),
        '평균매입단가':st.column_config.NumberColumn('평균매입단가',min_value=0.0,format='%.2f'),
        '현재가':st.column_config.NumberColumn('현재가',min_value=0.0,format='%.2f'),
        '환율(KRW/통화)':st.column_config.NumberColumn('환율(KRW/통화)',min_value=0.0,format='%.4f'),
    })
    col1,col2=st.columns(2)
    if col1.button('입력 내용 저장',use_container_width=True):
        st.session_state.holdings_data_v1=frame.copy().reset_index(drop=True)
        st.success('입력 내용이 현재 세션에 저장되었습니다.')
    if col2.button('현재가·환율 자동 조회',use_container_width=True):
        updated=frame.copy().reset_index(drop=True)
        errors=[]
        try:fx,fx_date=latest_fx()
        except Exception as e:fx,fx_date=None,'';errors.append(f'환율: {e}')
        for i,row in updated.iterrows():
            ticker=str(row.get('티커','')).strip().upper()
            if not ticker:continue
            if ticker in ('DEPOSIT','CMA'):
                errors.append(f'{ticker}: 자동 가격을 제공하지 않습니다. 수동 입력하세요.');continue
            try:
                price,currency,date=latest_quote(ticker)
                updated.at[i,'현재가']=price
                updated.at[i,'통화']=currency
                updated.at[i,'현재가 기준일']=date
                if currency=='KRW':updated.at[i,'환율(KRW/통화)']=1.0
                elif fx is not None:updated.at[i,'환율(KRW/통화)']=fx
            except Exception as e:errors.append(f'{ticker}: {str(e)[:100]}')
        st.session_state.holdings_data_v1=updated
        st.session_state.pop('holdings_editor_v1',None)
        if errors:st.warning('일부 조회 실패: '+' / '.join(errors[:8]))
        st.rerun()
    if st.button('보유자산 평가 및 PDF 생성',type='primary',use_container_width=True):
        try:
            data,summary=calculate_holdings(frame)
            st.session_state.holdings_result=(data,summary)
        except Exception as exc:st.error(f'입력값 확인: {exc}')
    if 'holdings_result' not in st.session_state:return
    data,summary=st.session_state.holdings_result
    st.caption('아래 결과는 마지막으로 분석 버튼을 누른 시점의 입력값입니다. 수정 후에는 다시 분석하세요.')
    c1,c2,c3=st.columns(3)
    c1.metric('총 평가금액',f"{summary['total_value']:,.0f}원")
    c2.metric('미실현 평가손익',f"{summary['unrealized_pnl']:+,.0f}원")
    c3.metric('원가 대비 평가수익률',f"{summary['unrealized_return']:.2%}" if np.isfinite(summary['unrealized_return']) else '산출 불가')
    st.dataframe(data[['종목명','티커','자산군','평가금액(KRW)','평가손익(KRW)','평가수익률','현재비중']],hide_index=True,use_container_width=True)
    chart=data.groupby('자산군',as_index=False)['평가금액(KRW)'].sum()
    st.plotly_chart(px.pie(chart,names='자산군',values='평가금액(KRW)',title='자산군별 평가금액 비중'),use_container_width=True)
    st.subheader('포트폴리오 진단')
    for paragraph in commentary(data,summary):st.write(paragraph)
    st.caption('Sharpe·MDD·VaR 등 기간별 위험지표는 과거 월별 수익률을 이용하는 백테스트 모드에서 확인하세요. 현재 잔고만으로 실제 운용성과를 산출하지 않습니다.')
    try:
        pdf=make_holdings_pdf(data,summary,dt.date.today().isoformat(),price_source='사용자 입력 또는 Yahoo Finance 최근 종가')
        st.download_button('📄 보유자산 리서치 PDF 다운로드',pdf,file_name='portfolio_lab_holdings_report.pdf',mime='application/pdf',type='primary')
    except Exception as exc:st.error(f'PDF 생성 실패: {exc}')

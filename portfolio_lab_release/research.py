"""Evidence-first, free research utilities. No generative AI or paywalled content."""
import re
from datetime import datetime, timezone
from urllib.parse import quote
import requests, feedparser
import pandas as pd

def ecos_series(key, stat_code, item_code, start='201801', end=None, cycle='M'):
    if not all([key,stat_code,item_code]): return pd.Series(dtype=float), 'ECOS 시계열 조회를 위해 인증키·통계표코드·항목코드를 입력하세요.'
    end=end or datetime.now(timezone.utc).strftime('%Y%m')
    url=f'https://ecos.bok.or.kr/api/StatisticSearch/{quote(key,safe="")}/json/kr/1/1000/{quote(stat_code,safe="")}/{cycle}/{start}/{end}/{quote(item_code,safe="")}'
    try:
        r=requests.get(url,timeout=22);r.raise_for_status();data=r.json()
        rows=data.get('StatisticSearch',{}).get('row',[])
        if not rows:return pd.Series(dtype=float), 'ECOS 통계자료 없음: 통계표·항목코드·주기를 확인하세요.'
        values=[]
        for item in rows:
            t=str(item.get('TIME',''));v=pd.to_numeric(item.get('DATA_VALUE'),errors='coerce')
            if cycle=='M' and re.fullmatch(r'\d{6}',t) and pd.notna(v):values.append((pd.to_datetime(t,format='%Y%m'),float(v)))
        if not values:return pd.Series(dtype=float),'월간 숫자 데이터가 없습니다.'
        s=pd.Series(dict(values)).sort_index();s.index.name='date';return s,None
    except Exception as e:return pd.Series(dtype=float),'ECOS 통계 조회 실패: '+str(e)[:160]

def kosis_series(key, org, table, item='ALL', obj='ALL', start='201801', end=None):
    if not all([key,org,table]):return pd.Series(dtype=float),'KOSIS 인증키·기관코드·통계표코드를 입력하세요.'
    end=end or datetime.now(timezone.utc).strftime('%Y%m')
    params={'method':'getList','apiKey':key,'format':'json','jsonVD':'Y','orgId':org,'tblId':table,'objL1':obj,'itmId':item,'prdSe':'M','startPrdDe':start,'endPrdDe':end}
    try:
        r=requests.get('https://kosis.kr/openapi/Param/statisticsParameterData.do',params=params,timeout=25);r.raise_for_status();data=r.json()
        if not isinstance(data,list):return pd.Series(dtype=float),'KOSIS 응답이 통계 데이터가 아닙니다.'
        items=[]
        for row in data:
            t=str(row.get('PRD_DE',''));v=pd.to_numeric(row.get('DT'),errors='coerce')
            if re.fullmatch(r'\d{6}',t) and pd.notna(v):items.append((pd.to_datetime(t,format='%Y%m'),float(v),str(row.get('ITM_NM','')),str(row.get('C1_NM',''))))
        distinct={(i[2],i[3]) for i in items}
        if len(distinct)>1:return pd.Series(dtype=float),f'KOSIS에 서로 다른 계열 {len(distinct)}개가 조회되었습니다. 항목·분류 코드를 구체적으로 선택하세요.'
        if not items:return pd.Series(dtype=float),'KOSIS 월간 숫자 데이터 없음: 표 주기·항목코드를 확인하세요.'
        s=pd.Series({t:v for t,v,*_ in items}).sort_index();s.index.name='date';return s,None
    except Exception as e:return pd.Series(dtype=float),'KOSIS 조회 실패: '+str(e)[:160]

def transform(s, mode):
    if mode=='전년동월비 (%)':return (s/s.shift(12)-1)*100
    if mode=='전월비 (%)':return (s/s.shift(1)-1)*100
    if mode=='3개월 이동평균':return s.rolling(3,min_periods=3).mean()
    return s.copy()

def interpret_series(s, label, mode, weight_fx=0):
    v=s.dropna()
    if len(v)<2:return '관측치가 부족하여 방향성 해석을 보류합니다.'
    recent=float(v.iloc[-1]);prior=float(v.iloc[-2]);delta=recent-prior
    direction='상승' if delta>0 else '하락' if delta<0 else '보합'
    # No causal claim from a single series.
    result=f'{label}: {v.index[-1]:%Y-%m} {recent:,.2f} (직전 관측치 대비 {delta:+,.2f}, {direction}). 지표 변환: {mode}. '
    if len(v)>=13:result+=f'12개월 전과 비교한 지표값 차이는 {recent-float(v.iloc[-13]):+,.2f}입니다. '
    result+='이는 관측된 통계적 변화이며, 원인과 향후 자산가격 방향을 단정하지 않습니다. '
    if weight_fx>0:result+=f'해외 표시 자산 비중 {weight_fx:.1f}%의 환율 위험과 함께 해석할 필요가 있습니다.'
    return result

def public_headlines(query, limit=5):
    # Google News RSS headlines/snippets only; not full article research.
    feed=feedparser.parse('https://news.google.com/rss/search?q='+quote(query)+'&hl=ko&gl=KR&ceid=KR:ko')
    result=[]
    for entry in feed.entries[:limit]:
        title=str(entry.get('title',''))[:160];url=str(entry.get('link',''))
        if not url.startswith('https://'):continue
        result.append({'title':title,'url':url,'published':str(entry.get('published','발행일 미확인'))})
    return result

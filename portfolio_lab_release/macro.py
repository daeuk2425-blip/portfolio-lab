"""Free public-statistics integration. No fabricated values or paid LLM calls."""
from datetime import date
import requests

ECOS_URL = 'https://ecos.bok.or.kr/api/'
KOSIS_URL = 'https://kosis.kr/openapi/statisticsData.do'

def ecos_key_statistics(key):
    if not key: return [], 'ECOS 인증키 미입력: 실시간 통계 조회 생략'
    url = f'{ECOS_URL}KeyStatisticList/{key}/json/kr/1/100'
    try:
        res=requests.get(url, timeout=18); res.raise_for_status()
        data=res.json(); block=data.get('KeyStatisticList', {})
        rows=block.get('row', [])
        if not isinstance(rows,list): rows=[]
        result=[]
        # These are real ECOS observations; do not infer a historical trend from one observation.
        for r in rows:
            name=str(r.get('KEYSTAT_NAME','')).strip()
            if any(word in name for word in ('기준금리','소비자물가','원/달러','원달러','환율','수출','실업률','산업생산')):
                value=str(r.get('DATA_VALUE','')).strip()
                if value not in ('','-','None'):
                    result.append({'name':name,'value':value,'unit':str(r.get('UNIT_NAME','')),
                        'period':str(r.get('CYCLE',''))+' '+str(r.get('TIME','')),
                        'source':'한국은행 ECOS','url':'https://ecos.bok.or.kr/'})
        return result[:12], (None if result else 'ECOS 응답에서 지정된 주요 지표를 찾지 못했습니다')
    except Exception as e:
        return [], 'ECOS 조회 실패: '+str(e)[:170]

def kosis_statistics(key, org_id, tbl_id):
    if not key or not org_id or not tbl_id:
        return [], 'KOSIS 인증키·기관코드·통계표코드가 없어 조회 생략'
    # Use a specific configured table. Never guess an item or silently use unrelated statistics.
    params={'method':'getList','apiKey':key,'format':'json','jsonVD':'Y',
            'orgId':org_id,'tblId':tbl_id,'prdSe':'M','newEstPrdCnt':'3',
            'itmId':'ALL','objL1':'ALL','objL2':'ALL'}
    try:
        res=requests.get(KOSIS_URL,params=params,timeout=18); res.raise_for_status()
        data=res.json()
        if not isinstance(data,list): return [], 'KOSIS API가 통계 행 대신 오류 또는 다른 응답을 반환했습니다'
        result=[]
        for r in data[:80]:
            if not isinstance(r,dict): continue
            val=str(r.get('DT','')).strip()
            if not val or val in ('-','...'): continue
            result.append({'name':str(r.get('ITM_NM','') or r.get('TBL_NM','') or tbl_id),
                'value':val,'unit':str(r.get('UNIT_NM','')),'period':str(r.get('PRD_DE','')),
                'source':'KOSIS '+str(tbl_id),'url':'https://kosis.kr/'})
        return result[:12], (None if result else 'KOSIS 통계값을 찾지 못했습니다. 통계표의 주기·항목 조건을 확인하세요')
    except Exception as e:
        return [], 'KOSIS 조회 실패: '+str(e)[:170]

def interpret_macro(allocation, stats, observations):
    tickers=[str(x).upper() for x in allocation['종목 코드']]
    weights=[float(x) for x in allocation['비중(%)']]
    overseas=sum(w for t,w in zip(tickers,weights) if t not in ('DEPOSIT','CMA') and not t.endswith(('.KS','.KQ')))
    domestic=sum(w for t,w in zip(tickers,weights) if t.endswith(('.KS','.KQ')))
    crypto=sum(w for t,w in zip(tickers,weights) if t in ('BTC-USD','ETH-USD'))
    largest=max(weights) if weights else 0
    parts=[f'최대 단일 자산 비중은 {largest:.1f}%입니다. 종목 간 상관관계와 공통 충격에 대한 집중도를 함께 점검해야 합니다.']
    if overseas: parts.append(f'해외 표시 자산 비중 {overseas:.1f}%는 원/달러 환율에 노출됩니다. 달러 강세는 원화 환산 수익률을 높일 수 있으나 반대의 경우 손실을 확대할 수 있습니다.')
    if domestic: parts.append(f'국내 주식 비중 {domestic:.1f}%는 국내 수요, 수출, 기업 실적과 관련될 수 있습니다. 산업별 실적 민감도는 종목별로 다릅니다.')
    if crypto: parts.append(f'암호화폐 비중 {crypto:.1f}%는 유동성 위축과 위험회피 심리에 민감할 수 있습니다. 주식과의 상관관계가 위기 때 상승할 가능성을 점검해야 합니다.')
    if observations:
        parts.append('아래 공공통계는 발표시점과 단위가 서로 다릅니다. 최신 관측치만으로 상승·하락 추세 또는 인과관계를 단정하지 않습니다.')
        parts.append('실제 관측치: '+'; '.join(f"{o['name']} {o['value']}{o['unit']} ({o['period']})" for o in observations[:4]))
    else:
        parts.append('공식 통계 실시간 수치를 확보하지 못해 거시경제의 현재 방향에 관한 판단은 유보합니다.')
    parts.append('분석적 시사점: 금리·물가·환율의 변화 방향을 확인한 뒤, 자산군별 민감도와 리밸런싱 거래비용을 함께 비교하는 접근이 적절합니다. 이는 시나리오이며 매매 지시가 아닙니다.')
    return ' '.join(parts)

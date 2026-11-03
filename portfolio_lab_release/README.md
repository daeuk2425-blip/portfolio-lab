# Portfolio Lab — 공개 배포 후보 버전

## 주요 기능
- 코스피/NASDAQ/S&P500/암호화폐 종목 검색 및 포트폴리오 편집
- 100% 비중 정규화, 5개 리밸런싱 전략 비교, 거래비용 가정
- 원화 환산 월간 백테스트, CAGR/Sharpe/Alpha/Beta/Treynor/VaR/MDD, 상관관계
- ECOS/KOSIS 시계열 조회 및 전월비/전년동월비/이동평균 가공 (공식 API 키 선택)
- 뉴스 RSS 제목과 원문 링크 (본문을 읽은 것으로 주장하지 않음)
- 정량 수치 기반 해석 문단 + 4페이지 애널리스트 스타일 PDF

## Colab
1. ZIP을 /content에 업로드하고 압축 해제
2. `apt-get -qq install -y fonts-nanum`
3. `pip install -r portfolio_lab_release/requirements.txt`
4. `python -m streamlit run portfolio_lab_release/app.py --server.port 8513 --server.address 0.0.0.0`
5. 기존 cloudflared로 localhost:8513 연결

## 상시 공개 배포 (Streamlit Community Cloud 등)
1. GitHub 새 저장소에 이 폴더 내 파일들을 저장소 루트에 업로드
2. Streamlit Community Cloud에서 GitHub 저장소 선택, entrypoint `app.py` 지정
3. 시스템 패키지로 Nanum 글꼴이 필요하므로 `packages.txt`를 유지
4. 앱 로그·가격 조회 실패·API 사용량·약관 확인 후 공개

## 중요한 한계
- 무료 외부 가격 서비스(yfinance) 및 상장종목 목록은 공개 상업 서비스에 대한 데이터 라이선스를 보장하지 않습니다. 배포 전 사용 조건 확인 또는 정식 데이터 공급자 교체가 필요합니다.
- 분석은 현재 구성 종목 기준의 과거 백테스트로 생존편향 가능성이 있습니다.
- 거래세, 슬리피지, 세금, 예금 중도해지 및 배당 원천징수는 미반영입니다.
- 'AI 애널리스트'를 사칭하지 않습니다. 규칙 기반 문단이며 공개 리서치 원문 자동 해석은 아직 없습니다.
- Colab/임시 터널은 24시간 서비스 운영에 적합하지 않습니다.
- 개인별 포트폴리오 저장/로그인/계정 관리 기능은 포함하지 않습니다.
- API 키는 코드에 넣거나 공개 저장소에 커밋하지 마세요.

# 디스코드 리포트 출력 명세

## 1. 리포트 생성 규칙
- **필터링:** 종합 점수 상위 10~50위 종목만 출력.
- **빈도:** 분석 엔진 가동 직후 자동 전송 (또는 사용자 요청 시).

## 2. 출력 항목 (Template)
- **종목명 & 현재가:** (예: ETH/USDT @ $2,450)
- **전략 방향:** LONG / SHORT
- **핵심 근거:** (예: Wyckoff Phase C + 4h FVG + Whale Netflow Positive)
- **매매 가이드:**
    - **Entry:** 특정 가격대 (Range)
    - **Stop Loss:** MSS 무효화 지점
    - **Target 1:** 리스크 관리 구간 (R:R 1:1)
    - **Target 2:** 주요 매물대 (R:R 1:2)
    - **Target 3:** 추세 최대치 (R:R 1:3+)
- **Visuals:** 에이전트가 생성한 주요 지표 요약 테이블 첨부.
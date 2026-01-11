# 파이프라인 및 데이터 핸들링 명세

## 1. 수집 대상 및 주기
- **대상:** 시총 1위 ~ 200위 (CoinMarketCap 기준 매일 업데이트).
- **주기:** - 1h, 4h: 1시간마다 1회 수집.
    - 1D, 1W: 1일 1회 수집.
- **포맷:** Pandas DataFrame (Open, High, Low, Close, Volume, OI, Funding Rate, CVD).

## 2. API 이중화 및 속도 제한(Rate Limit) 대응
- **Primary:** Binance API (Spot & Futures).
- **Secondary:** Bybit API (Binance 장애 시 즉시 스위칭).
- **Parallel Processing:** 200개 종목 분석 시 `asyncio` 또는 `multiprocessing`을 사용하여 수집 시간을 5분 이내로 단축.
- **Batching:** API 요청 시 20개 단위로 묶어 호출하여 Rate Limit 초과 방지.
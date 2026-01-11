# 분석 알고리즘 로직 명세

## 1. ICT & Wyckoff 감지 로직
- **MSS (Market Structure Shift):** 이전 하락 추세의 고점(HH)을 상향 돌파하는 첫 번째 캔들 포착.
- **FVG (Fair Value Gap):** 3개의 연속된 캔들 중 1번 캔들의 고점과 3번 캔들의 저점 사이의 공백 확인.
- **Wyckoff Phase C (Spring):** 주요 지지선을 일시적으로 하향 이탈한 후, 대량의 거래량(또는 Whale Inflow)과 함께 복귀하는 구간 식별.

## 2. VCP (Volatility Contraction Pattern) 로직
- **Tightness:** 캔들의 변동폭(High-Low)이 이전 5개 캔들의 평균 변동폭의 50% 이하로 축소되는 지점 감지.
- **Volume Dry-up:** 돌파 직전 거래량이 급감하는 현상을 온체인 지표와 결합하여 분석.

## 3. 온체인 결합 필터
- **조건:** ICT Buy Setup 발생 시, **CVD가 상승 중이거나 OI가 증가**하고 있다면 'High Conviction' 부여.
- **조건:** 가격은 오르나 **고래 유입이 없고 펀딩비만 급증**한다면 'Fake-out' 경고 생성.
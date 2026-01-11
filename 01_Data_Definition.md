# 데이터 정의 및 우선순위 명세

## 1. 핵심 온체인 지표 (On-chain Confluence)
에이전트는 다음 지표를 '기관의 의도'로 해석한다:
- **Whale Net Flow:** 거래소 지갑으로의 대규모 스테이블코인 유입(Inflow)은 잠재적 매수세로, 코인 유출(Outflow)은 매집 후 보관으로 해석.
- **Liquidation Heatmap:** 특정 가격대(ICT의 Liquidity Pool)에 몰려 있는 청산 물량 규모 확인.
- **Stablecoin Supply Ratio (SSR):** 시장의 구매력 지수로 활용.
- **Mean Coin Age:** 장기 홀더들의 매집 혹은 분산 국면 판단.

## 2. 시장 지표 (Market Indicators)
- **CVD (Cumulative Volume Delta):** 가격 상승 시 CVD 하락은 '흡수(Absorption)' 패턴으로 감지.
- **Open Interest (OI):** 가격 돌파 시 OI 동반 상승 여부로 '진성 돌파' 판별.
- **Funding Rate:** 0.01% 이상 과열 시 롱 스퀴즈, 음수 전환 시 숏 스퀴즈 위험 가중치 부여.
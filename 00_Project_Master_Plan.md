# 프로젝트: Antigravity-Alpha 통합 설계도

## 1. 시스템 비전
본 프로젝트는 온체인 데이터(스마트 머니의 흐름)와 고급 기술적 분석(ICT/Wyckoff/VCP)을 결합하여, 시총 200위권 내 최적의 매매 기회를 자동으로 포착하고 디스코드로 보고하는 '에이전트 기반 퀀트 시스템'이다.

## 2. 모듈별 역할 (Modular Architecture)
1. **Pipeline (MD-02):** 데이터 수집 및 정규화. 200개 종목의 멀티 타임프레임 데이터 확보.
2. **Analysis Engine (MD-03):** ICT, Wyckoff, VCP 알고리즘을 통한 기계적 패턴 식별.
3. **Reasoning Agent (MD-04):** LLM이 기술 데이터와 온체인 데이터를 결합하여 최종 전략 도출.
4. **Distributor (MD-05):** 최상위 10~50개 종목을 선별하여 디스코드 리포트 송출.

## 3. 에이전트 워크플로우 (Flow)
1. [Data Agent] -> 200개 종목 데이터 수집 및 DB 저장.
2. [Logic Agent] -> 저장된 데이터에서 패턴(MSS, FVG, VCP 등) 검색 및 스코어링.
3. [CIO Agent] -> 온체인 데이터를 대조하여 필터링 및 최종 Entry/TP/SL 확정.
4. [Report Agent] -> 디스코드 API를 호출하여 시각화된 보고서 전송.
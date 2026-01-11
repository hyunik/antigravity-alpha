# 🚀 Antigravity-Alpha: Smart Coin Analysis System

AI 기반 암호화폐 자동 분석 시스템. 6가지 핵심 분석(HTF/LTF, Valuation, On-Chain, VPA, ICT, Wyckoff)을 통합하여 매매 추천을 생성하고 Discord로 알림을 전송합니다.

## ✨ 주요 기능

- **📊 6요소 종합 분석**: HTF/LTF 탑다운, Valuation, On-Chain, VPA, ICT, Wyckoff
- **🤖 LLM 추론**: OpenAI GPT-4o 또는 Google Gemini 선택 가능
- **📡 자동 알림**: Discord Webhook으로 실시간 알림
- **📈 성과 추적**: 추천 결과 자동 추적 및 승률/P&L 분석
- **⏰ 자동 실행**: GitHub Actions로 매일/매주 자동 분석

## 🏃 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/antigravity-alpha.git
cd antigravity-alpha

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 입력
```

### 3. 실행

```bash
# 단일 코인 빠른 분석
python main.py --mode quick --symbol BTCUSDT

# Top 100 코인 전체 분석
python main.py --mode full --limit 100

# 주간 성과 보고서
python weekly_report.py --days 30
```

## 📋 GitHub Actions 설정

### 1. Repository Secrets 설정

GitHub Repository > Settings > Secrets and variables > Actions에서 다음 시크릿 추가:

| Secret Name | Description |
|-------------|-------------|
| `LLM_PROVIDER` | `openai` 또는 `gemini` |
| `OPENAI_API_KEY` | OpenAI API 키 |
| `GEMINI_API_KEY` | Google Gemini API 키 |
| `DISCORD_WEBHOOK_URL` | Discord Webhook URL |

### 2. 워크플로우 일정

| 워크플로우 | 실행 시간 | 설명 |
|-----------|----------|------|
| Daily Analysis | 매일 09:00 KST | Top 100 코인 분석 및 Discord 알림 |
| Weekly Report | 매주 월요일 18:00 KST | 주간 성과 보고서 Discord 전송 |

### 3. 수동 실행

GitHub Actions 탭에서 "Run workflow" 버튼으로 수동 실행 가능

## 📁 프로젝트 구조

```
antigravity-alpha/
├── .github/workflows/     # GitHub Actions
│   ├── daily_analysis.yml # 일일 분석
│   └── weekly_report.yml  # 주간 보고서
├── analysis/              # 분석 엔진
│   ├── comprehensive_analyzer.py  # 6요소 통합 분석
│   ├── ict_detector.py    # ICT 패턴 감지
│   ├── wyckoff_detector.py # Wyckoff 분석
│   ├── vpa_detector.py    # VPA 분석
│   └── performance_tracker.py # 성과 추적
├── pipeline/              # 데이터 수집
├── agents/                # LLM 에이전트
├── distribution/          # Discord 전송
├── main.py               # 메인 실행 파일
├── scheduler.py          # 스케줄러
└── weekly_report.py      # 주간 보고서
```

## 📊 보고서 예시

### 종합 분석 보고서
```
📊 종합 분석 보고서 : BTCUSDT

[최종 분석 요약]
• HTF/LTF: 상승 정렬
• Valuation: 유통률 93%, 언락 리스크 미미
• On-Chain: 롱 포지션 증가
• VPA: 매수 신호 우세
• ICT: FVG 상승 셋업
• Wyckoff: Markup Phase

[트레이딩 전략]
• Entry: $89,000 - $90,000
• TP: $92,000 / $95,000 / $100,000
• SL: $87,000
```

### 성과 보고서
```
📈 성과 분석 보고서 (최근 30일)

승률: 73.3% | Profit Factor: 3.33x
총 P&L: +113.41% | 평균 P&L: +7.56%

신뢰도별 성과:
HIGH: 100% 승률 | MEDIUM: 83% 승률 | LOW: 50% 승률
```

## 🔧 개발

```bash
# 테스트 실행
python -m pytest tests/

# 타입 체크
python -m mypy .
```

## ⚠️ 면책 조항

이 시스템은 교육 및 연구 목적으로 제작되었습니다. 
실제 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
암호화폐 투자는 원금 손실의 위험이 있습니다.

## 📄 라이선스

MIT License

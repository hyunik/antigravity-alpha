"""
Comprehensive Report Generator
Generates detailed analysis reports with 6 core elements
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from analysis.htf_ltf_analyzer import HTFLTFAnalyzer, TrendDirection
from analysis.vpa_detector import VPADetector
from analysis.valuation_analyzer import ValuationAnalyzer, ValuationData
from analysis.ict_detector import ICTDetector
from analysis.wyckoff_detector import WyckoffDetector
from analysis.scoring_engine import ScoringEngine, CoinScore


@dataclass
class ComprehensiveReport:
    """Comprehensive analysis report"""
    symbol: str
    timestamp: str
    
    # 6 Core Analysis Scores
    htf_ltf_score: float
    valuation_score: float
    onchain_score: float
    vpa_score: float
    ict_score: float
    wyckoff_score: float
    
    # Weighted Total Score
    total_score: float
    
    # Analysis Summaries
    htf_ltf_summary: str
    valuation_summary: str
    onchain_summary: str
    vpa_summary: str
    ict_summary: str
    wyckoff_summary: str
    
    # Consensus & Conflict
    bullish_factors: List[str]
    bearish_factors: List[str]
    consensus: str
    conflict_resolution: str
    
    # Final Verdict
    final_bias: str  # "매수 우위", "매도 우위", "중립"
    conviction: str  # "HIGH", "MEDIUM", "LOW"
    
    # Trading Plan
    recommended_action: str
    entry_price: float
    entry_zone: Tuple[float, float]
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_ratio: float
    leverage_recommendation: str


class ComprehensiveAnalyzer:
    """
    Comprehensive Analysis Engine
    
    Integrates all 6 core analysis elements:
    1. HTF/LTF Top-Down Analysis
    2. Valuation (Mcap, FDV, Unlock)
    3. On-Chain (CVD, OI, Funding Rate)
    4. VPA (Volume Price Analysis)
    5. ICT (Inner Circle Trader)
    6. Wyckoff Theory
    """
    
    # Weights for each analysis component
    WEIGHTS = {
        "htf_ltf": 0.20,      # 20%
        "valuation": 0.10,    # 10%
        "onchain": 0.20,      # 20%
        "vpa": 0.15,          # 15%
        "ict": 0.20,          # 20%
        "wyckoff": 0.15       # 15%
    }
    
    def __init__(self):
        self.htf_ltf_analyzer = HTFLTFAnalyzer()
        self.vpa_detector = VPADetector()
        self.valuation_analyzer = ValuationAnalyzer()
        self.ict_detector = ICTDetector()
        self.wyckoff_detector = WyckoffDetector()
    
    def analyze_onchain(
        self,
        open_interest: float,
        funding_rate: float,
        oi_change_pct: float = 0,
        price_change_pct: float = 0
    ) -> Dict:
        """
        Analyze on-chain/market data
        
        Args:
            open_interest: Current OI
            funding_rate: Current funding rate
            oi_change_pct: OI change in last 24h
            price_change_pct: Price change in last 24h
        """
        score = 50
        signals = []
        
        # Funding Rate Analysis
        if funding_rate > 0.01:
            score -= 15
            signals.append(f"⚠️ 과열된 롱 포지션 (펀딩비 {funding_rate*100:.3f}%, 롱 스퀴즈 주의)")
        elif funding_rate > 0.005:
            score -= 5
            signals.append(f"롱 우위 시장 (펀딩비 {funding_rate*100:.3f}%)")
        elif funding_rate < -0.01:
            score += 15
            signals.append(f"숏 과열 (펀딩비 {funding_rate*100:.3f}%, 숏 스퀴즈 가능)")
        elif funding_rate < -0.005:
            score += 5
            signals.append(f"숏 우위 시장 (펀딩비 {funding_rate*100:.3f}%)")
        else:
            signals.append(f"중립적 펀딩비 ({funding_rate*100:.3f}%)")
        
        # OI + Price divergence
        if oi_change_pct > 10 and price_change_pct > 5:
            score += 10
            signals.append("신규 롱 포지션 공격적 유입 (OI↑ + 가격↑)")
        elif oi_change_pct > 10 and price_change_pct < -5:
            score -= 10
            signals.append("신규 숏 포지션 공격적 유입 (OI↑ + 가격↓)")
        elif oi_change_pct < -10 and price_change_pct > 5:
            signals.append("숏 포지션 청산 (OI↓ + 가격↑)")
        elif oi_change_pct < -10 and price_change_pct < -5:
            signals.append("롱 포지션 청산 (OI↓ + 가격↓)")
        
        # Crowding Risk
        crowding_risk = funding_rate > 0.008 or funding_rate < -0.008
        if crowding_risk:
            signals.append("⚠️ 과열 경고: 스퀴즈 가능성 상존")
        
        summary = " | ".join(signals[:3])
        
        return {
            "onchain_score": max(0, min(100, score)),
            "summary": summary,
            "signals": signals,
            "crowding_risk": crowding_risk,
            "funding_rate": funding_rate,
            "oi": open_interest
        }
    
    def generate_trading_plan(
        self,
        current_price: float,
        direction: str,
        support: float,
        resistance: float,
        atr: float
    ) -> Dict:
        """Generate specific trading plan with entry, SL, TP"""
        
        if direction in ["LONG", "매수"]:
            # Entry near support
            entry_low = support
            entry_high = support + atr * 0.5
            entry = (entry_low + entry_high) / 2
            
            # Stop loss below support
            sl = support - atr * 0.5
            
            # Targets
            risk = entry - sl
            tp1 = entry + risk * 1.0  # 1:1 R:R
            tp2 = entry + risk * 2.0  # 1:2 R:R
            tp3 = min(entry + risk * 3.0, resistance)  # 1:3 or resistance
            
            rr = (tp2 - entry) / (entry - sl) if (entry - sl) > 0 else 0
            
        else:  # SHORT
            # Entry near resistance
            entry_low = resistance - atr * 0.5
            entry_high = resistance
            entry = (entry_low + entry_high) / 2
            
            # Stop loss above resistance
            sl = resistance + atr * 0.5
            
            # Targets
            risk = sl - entry
            tp1 = entry - risk * 1.0
            tp2 = entry - risk * 2.0
            tp3 = max(entry - risk * 3.0, support)
            
            rr = (entry - tp2) / (sl - entry) if (sl - entry) > 0 else 0
        
        return {
            "entry": entry,
            "entry_zone": (entry_low, entry_high),
            "stop_loss": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "risk_reward": rr
        }
    
    def analyze(
        self,
        symbol: str,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
        open_interest: float = 0,
        funding_rate: float = 0,
        market_cap: float = 0,
        fdv: float = 0,
        circulating_supply: float = 0,
        total_supply: float = 0
    ) -> ComprehensiveReport:
        """
        Perform comprehensive analysis with all 6 elements
        
        Returns:
            ComprehensiveReport with full analysis
        """
        current_price = df_4h["close"].iloc[-1] if df_4h is not None and len(df_4h) > 0 else 0
        
        # 1. HTF/LTF Analysis
        htf_ltf_result = self.htf_ltf_analyzer.analyze(
            df_1h=df_1h, df_4h=df_4h, df_1d=df_1d
        )
        htf_ltf_score = htf_ltf_result.get("htf_ltf_score", 50)
        htf_ltf_summary = htf_ltf_result.get("summary", "분석 불가")
        
        # 2. Valuation Analysis
        if total_supply > 0:
            valuation_data = self.valuation_analyzer.analyze_from_market_data(
                symbol, market_cap, fdv, circulating_supply, total_supply
            )
            valuation_score = valuation_data.valuation_score
            valuation_summary = valuation_data.summary
        else:
            valuation_score = 50
            valuation_summary = "밸류에이션 데이터 없음"
        
        # 3. On-Chain Analysis
        onchain_result = self.analyze_onchain(open_interest, funding_rate)
        onchain_score = onchain_result["onchain_score"]
        onchain_summary = onchain_result["summary"]
        
        # 4. VPA Analysis
        vpa_df = df_4h if df_4h is not None else df_1h
        if vpa_df is not None and len(vpa_df) > 30:
            vpa_result = self.vpa_detector.analyze(vpa_df)
            vpa_score = vpa_result["vpa_score"]
            vpa_summary = vpa_result["summary"]
        else:
            vpa_score = 50
            vpa_summary = "VPA 데이터 부족"
        
        # 5. ICT Analysis
        ict_df = df_4h if df_4h is not None else df_1h
        if ict_df is not None and len(ict_df) > 20:
            ict_result = self.ict_detector.analyze(ict_df)
            ict_score = ict_result["ict_score"]
            ict_bias = ict_result["bias"]
            ict_summary = f"ICT Bias: {ict_bias} | MSS: {ict_result['has_mss']} | FVG: {ict_result['has_fvg']}"
        else:
            ict_score = 50
            ict_summary = "ICT 데이터 부족"
        
        # 6. Wyckoff Analysis
        wyckoff_df = df_1d if df_1d is not None and len(df_1d) > 30 else df_4h
        if wyckoff_df is not None and len(wyckoff_df) > 20:
            wyckoff_result = self.wyckoff_detector.analyze(wyckoff_df)
            wyckoff_score = wyckoff_result.wyckoff_score
            wyckoff_summary = f"Phase: {wyckoff_result.phase.value} | Spring: {wyckoff_result.has_spring}"
        else:
            wyckoff_score = 50
            wyckoff_summary = "Wyckoff 데이터 부족"
        
        # Calculate Weighted Total Score
        total_score = (
            htf_ltf_score * self.WEIGHTS["htf_ltf"] +
            valuation_score * self.WEIGHTS["valuation"] +
            onchain_score * self.WEIGHTS["onchain"] +
            vpa_score * self.WEIGHTS["vpa"] +
            ict_score * self.WEIGHTS["ict"] +
            wyckoff_score * self.WEIGHTS["wyckoff"]
        )
        
        # Collect bullish and bearish factors
        bullish_factors = []
        bearish_factors = []
        
        if htf_ltf_score >= 60:
            bullish_factors.append(f"HTF/LTF 상승 정렬 (점수: {htf_ltf_score})")
        elif htf_ltf_score <= 40:
            bearish_factors.append(f"HTF/LTF 하락 정렬 (점수: {htf_ltf_score})")
        
        if valuation_score >= 70:
            bullish_factors.append(f"건강한 토크노믹스 (유통률 높음)")
        
        if onchain_score >= 60:
            bullish_factors.append(f"수급 양호 ({onchain_summary})")
        elif onchain_score <= 40:
            bearish_factors.append(f"수급 악화 ({onchain_summary})")
        
        if vpa_score >= 60:
            bullish_factors.append("VPA 매수 신호")
        elif vpa_score <= 40:
            bearish_factors.append("VPA 매도 신호 (클라이맥스/소진)")
        
        if ict_score >= 60:
            bullish_factors.append(f"ICT 매수 셋업 ({ict_summary})")
        elif ict_score <= 40:
            bearish_factors.append(f"ICT 매도 셋업")
        
        if wyckoff_score >= 60:
            bullish_factors.append(f"Wyckoff 매집/상승 ({wyckoff_summary})")
        elif wyckoff_score <= 40:
            bearish_factors.append(f"Wyckoff 분산/하락")
        
        # Consensus & Conflict Resolution
        if len(bullish_factors) > len(bearish_factors) + 2:
            consensus = "강한 매수 합의"
            final_bias = "매수 우위"
        elif len(bearish_factors) > len(bullish_factors) + 2:
            consensus = "강한 매도 합의"
            final_bias = "매도 우위"
        elif len(bullish_factors) > len(bearish_factors):
            consensus = "약한 매수 우위"
            final_bias = "매수 우위"
        elif len(bearish_factors) > len(bullish_factors):
            consensus = "약한 매도 우위"
            final_bias = "매도 우위"
        else:
            consensus = "중립 (혼합 신호)"
            final_bias = "중립"
        
        conflict_resolution = ""
        if bullish_factors and bearish_factors:
            conflict_resolution = (
                f"상충 신호 존재: 강세({len(bullish_factors)}개) vs 약세({len(bearish_factors)}개). "
                f"{'추세를 따르되 조정 시 진입 권장' if final_bias == '매수 우위' else '관망 또는 역추세 진입 시 신중'}"
            )
        
        # Conviction level
        if total_score >= 75:
            conviction = "HIGH"
        elif total_score >= 55:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"
        
        # Trading Plan
        if df_4h is not None and len(df_4h) > 20:
            atr = (df_4h["high"] - df_4h["low"]).tail(14).mean()
            support = wyckoff_result.support_level if wyckoff_df is not None else df_4h["low"].tail(20).min()
            resistance = wyckoff_result.resistance_level if wyckoff_df is not None else df_4h["high"].tail(20).max()
            
            direction = "LONG" if final_bias == "매수 우위" else "SHORT"
            plan = self.generate_trading_plan(current_price, direction, support, resistance, atr)
        else:
            plan = {
                "entry": current_price,
                "entry_zone": (current_price * 0.98, current_price),
                "stop_loss": current_price * 0.95,
                "tp1": current_price * 1.05,
                "tp2": current_price * 1.10,
                "tp3": current_price * 1.15,
                "risk_reward": 2.0
            }
        
        # Leverage recommendation
        if conviction == "HIGH" and final_bias != "중립":
            leverage_rec = "중위험 (2-5x)"
        elif conviction == "MEDIUM":
            leverage_rec = "저위험 (1-2x)"
        else:
            leverage_rec = "무레버리지 권장"
        
        # Recommended action
        if final_bias == "매수 우위" and total_score >= 60:
            if total_score >= 75:
                recommended_action = "적극 매수"
            else:
                recommended_action = "조정 시 분할 매수"
        elif final_bias == "매도 우위" and total_score <= 40:
            recommended_action = "매도 우위 / 숏 진입 고려"
        else:
            recommended_action = "관망 또는 진입 대기"
        
        return ComprehensiveReport(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            htf_ltf_score=htf_ltf_score,
            valuation_score=valuation_score,
            onchain_score=onchain_score,
            vpa_score=vpa_score,
            ict_score=ict_score,
            wyckoff_score=wyckoff_score,
            total_score=total_score,
            htf_ltf_summary=htf_ltf_summary,
            valuation_summary=valuation_summary,
            onchain_summary=onchain_summary,
            vpa_summary=vpa_summary,
            ict_summary=ict_summary,
            wyckoff_summary=wyckoff_summary,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            consensus=consensus,
            conflict_resolution=conflict_resolution,
            final_bias=final_bias,
            conviction=conviction,
            recommended_action=recommended_action,
            entry_price=plan["entry"],
            entry_zone=plan["entry_zone"],
            stop_loss=plan["stop_loss"],
            take_profit_1=plan["tp1"],
            take_profit_2=plan["tp2"],
            take_profit_3=plan["tp3"],
            risk_reward_ratio=plan["risk_reward"],
            leverage_recommendation=leverage_rec
        )


def format_comprehensive_report(report: ComprehensiveReport) -> str:
    """Format comprehensive report as detailed narrative text"""
    
    # Generate narrative descriptions for each element
    htf_ltf_narrative = _generate_htf_ltf_narrative(report)
    valuation_narrative = _generate_valuation_narrative(report)
    onchain_narrative = _generate_onchain_narrative(report)
    vpa_narrative = _generate_vpa_narrative(report)
    ict_narrative = _generate_ict_narrative(report)
    wyckoff_narrative = _generate_wyckoff_narrative(report)
    executive_summary = _generate_executive_summary(report)
    consensus_narrative = _generate_consensus_narrative(report)
    trading_plan_narrative = _generate_trading_plan_narrative(report)
    
    output = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 종합 분석 보고서 : {report.symbol}
⏰ 분석 시간 : {report.timestamp[:16].replace('T', ' ')} UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


━━━━━━━━━━━━━━━━ [1] 최종 분석 요약 (Executive Summary) ━━━━━━━━━━━━━━━━

{executive_summary}

현재 시장은 {_get_market_condition(report)}. 

최종 판단은 「{report.final_bias}」입니다. {_get_final_judgment_reason(report)}


━━━━━━━━━━━━━━━━ [2] 6대 핵심 분석 상세 ━━━━━━━━━━━━━━━━

📈 【HTF/LTF 탑다운 분석】 (점수: {report.htf_ltf_score:.0f}/100)
{htf_ltf_narrative}

💰 【밸류에이션 분석】 (점수: {report.valuation_score:.0f}/100)
{valuation_narrative}

🔗 【온체인 데이터 분석】 (점수: {report.onchain_score:.0f}/100)
{onchain_narrative}

📊 【VPA (거래량-가격 분석)】 (점수: {report.vpa_score:.0f}/100)
{vpa_narrative}

🎯 【ICT (스마트 머니 분석)】 (점수: {report.ict_score:.0f}/100)
{ict_narrative}

🔄 【Wyckoff 시장 국면 분석】 (점수: {report.wyckoff_score:.0f}/100)
{wyckoff_narrative}


━━━━━━━━━━━━━━━━ [3] 합의(Consensus) 및 충돌(Conflict) 해결 ━━━━━━━━━━━━━━━━

{consensus_narrative}


━━━━━━━━━━━━━━━━ [4] 종합 스코어링 (가중 평균) ━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────┐
│ 분석 항목               │ 점수      │ 가중치   │
├─────────────────────────────────────────────────┤
│ 기술적 구조 (HTF/LTF)   │ {report.htf_ltf_score:5.0f}점   │   20%    │
│ 펀더멘탈 (Valuation)    │ {report.valuation_score:5.0f}점   │   10%    │
│ 수급/유동성 (On-Chain)  │ {report.onchain_score:5.0f}점   │   20%    │
│ VPA (거래량 분석)       │ {report.vpa_score:5.0f}점   │   15%    │
│ ICT (스마트 머니)       │ {report.ict_score:5.0f}점   │   20%    │
│ Wyckoff (시장 국면)     │ {report.wyckoff_score:5.0f}점   │   15%    │
├─────────────────────────────────────────────────┤
│ ▶ 최종 종합 점수        │ {report.total_score:5.0f}점   │  100%    │
└─────────────────────────────────────────────────┘

점수 해석: 0~40점 = 매도 우위 / 41~60점 = 중립 / 61~100점 = 매수 우위


━━━━━━━━━━━━━━━━ [5] 최종 트레이딩 전략 (Action Plan) ━━━━━━━━━━━━━━━━

{trading_plan_narrative}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return output


def _generate_htf_ltf_narrative(report: ComprehensiveReport) -> str:
    """Generate HTF/LTF narrative"""
    score = report.htf_ltf_score
    summary = report.htf_ltf_summary
    
    if "상승 정렬" in summary or score >= 70:
        return f"""상위 타임프레임(HTF)과 하위 타임프레임(LTF)이 모두 상승 추세를 
보이고 있습니다. 이는 시장의 큰 흐름과 세부 타점이 일치하는 이상적인 
상황으로, 추세 추종 매매에 유리한 환경입니다. {summary}"""
    
    elif "하락 정렬" in summary or score <= 30:
        return f"""상위 타임프레임(HTF)과 하위 타임프레임(LTF)이 모두 하락 추세를 
보이고 있습니다. 시장 전반에 하방 압력이 존재하며, 롱 포지션 진입 시 
각별한 주의가 필요합니다. {summary}"""
    
    elif "역행" in summary:
        return f"""현재 상위 프레임(HTF)과 하위 프레임(LTF) 간 방향성이 다른 '역행 구조'가 
나타나고 있습니다. {summary}
        
이런 상황에서는 HTF 방향으로의 회귀 가능성을 염두에 두고, LTF 움직임은 
단기 기회 또는 조정으로 해석하는 것이 적절합니다. 역추세 매매 시 
손절 관리를 철저히 해야 합니다."""
    
    else:
        return f"""현재 타임프레임 간 명확한 추세 정렬이 이루어지지 않은 상태입니다.
{summary}
        
방향성이 불분명할 때는 진입을 자제하고, 추세가 명확해질 때까지 
관망하는 것이 안전합니다."""


def _generate_valuation_narrative(report: ComprehensiveReport) -> str:
    """Generate valuation narrative"""
    score = report.valuation_score
    summary = report.valuation_summary
    
    if score >= 90:
        return f"""토크노믹스 측면에서 매우 안정적인 구조를 보입니다. {summary}
        
완전 희석 가치(FDV)와 시가총액(MCap)의 비율이 낮아 향후 토큰 언락으로 인한 
매도 압력 우려가 적습니다. 펀더멘탈 측면에서 추세를 지지할 체력이 
충분하다고 판단됩니다."""
    
    elif score >= 70:
        return f"""밸류에이션이 양호한 편입니다. {summary}
        
유통률이 높아 대규모 언락 이벤트로 인한 급격한 공급 증가 위험은 
제한적입니다. 다만 향후 언락 스케줄은 별도로 확인하시기 바랍니다."""
    
    elif score >= 50:
        return f"""밸류에이션이 보통 수준입니다. {summary}
        
일부 미유통 물량이 존재하므로 향후 언락 일정을 확인하고, 
대량 언락 전후로는 변동성 확대에 주의해야 합니다."""
    
    else:
        return f"""밸류에이션 측면에서 리스크가 존재합니다. {summary}
        
⚠️ 높은 미유통 비율은 향후 대량 언락 시 매도 압력으로 작용할 수 있으므로, 
언락 스케줄을 반드시 확인하고 투자에 신중을 기해야 합니다."""


def _generate_onchain_narrative(report: ComprehensiveReport) -> str:
    """Generate on-chain narrative"""
    score = report.onchain_score
    summary = report.onchain_summary
    
    if score >= 70:
        return f"""온체인 지표가 매수세 우위를 시사하고 있습니다. {summary}
        
펀딩비와 미결제약정(OI) 추이를 종합하면, 시장 참여자들이 적극적으로 
롱 포지션을 구축하고 있으며 이는 단기 상승 모멘텀을 지지합니다."""
    
    elif score >= 55:
        return f"""온체인 데이터가 비교적 중립적인 상태입니다. {summary}
        
펀딩비가 안정적이어서 극단적인 포지션 쏠림은 관찰되지 않습니다.
다만 신규 자금 유입 강도는 제한적이어서, 강한 추세 발생을 위해서는 
추가적인 수급 개선이 필요합니다."""
    
    elif score >= 40:
        return f"""온체인 지표에서 주의 신호가 포착됩니다. {summary}
        
현재 상태는 과열 또는 스퀴즈 가능성이 공존하는 경계 구간입니다.
포지션 방향에 따른 스퀴즈 리스크를 반드시 고려해야 합니다."""
    
    else:
        return f"""온체인 데이터가 약세를 시사합니다. {summary}
        
⚠️ 펀딩비 또는 OI 추이에서 부정적 신호가 관찰됩니다. 
현재 수급 상황은 추세 전환 또는 급격한 변동을 예고할 수 있으므로, 
레버리지 사용을 자제하고 리스크 관리를 철저히 해야 합니다."""


def _generate_vpa_narrative(report: ComprehensiveReport) -> str:
    """Generate VPA narrative"""
    score = report.vpa_score
    summary = report.vpa_summary
    
    if "클라이맥스" in summary or "소진" in summary:
        return f"""VPA(Volume Price Analysis) 분석 결과, 추세 소진 신호가 포착되었습니다.
{summary}

거래량이 급증하며 상승 클라이맥스(Buying Climax) 또는 하락 클라이맥스 
(Selling Climax) 패턴이 다수 출현했습니다. 이는 단기 과열 상태를 의미하며, 
추격 매수/매도보다는 조정을 기다리는 것이 바람직합니다."""
    
    elif score >= 65:
        return f"""VPA 분석 결과, 매수 신호가 우세합니다. {summary}

거래량과 가격의 상호작용이 매수세 강화를 시사하며, 
'흡수(Absorption)' 또는 '스프링(Spring)' 패턴이 관찰됩니다. 
이는 스마트 머니의 매집 활동을 암시할 수 있습니다."""
    
    elif score >= 35:
        return f"""VPA 분석 결과, 중립적인 상태입니다. {summary}

거래량과 가격 움직임이 명확한 방향성을 제시하지 않고 있습니다.
추세 발생 시 거래량 동반 여부를 확인하여 진위를 판별해야 합니다."""
    
    else:
        return f"""VPA 분석 결과, 매도 신호가 우세합니다. {summary}

거래량 분석에서 'No Demand(무수요)' 또는 'Upthrust(업스러스트)' 패턴이 
관찰되어, 상승 시도 시 매도 압력을 받을 가능성이 높습니다.
롱 포지션 진입 시 신중해야 합니다."""


def _generate_ict_narrative(report: ComprehensiveReport) -> str:
    """Generate ICT narrative"""
    score = report.ict_score
    summary = report.ict_summary
    
    if score >= 70:
        return f"""ICT(Inner Circle Trader) 분석 결과, 스마트 머니의 매수 활동이 감지됩니다.
{summary}

시장 구조 변화(MSS)와 공정가치갭(FVG)이 상승 방향으로 정렬되어 있으며,
이는 기관 트레이더들이 매수 포지션을 구축하고 있음을 시사합니다.
현재 가격대는 할인(Discount) 영역에 해당할 수 있습니다."""
    
    elif score >= 40:
        return f"""ICT 분석 결과, 혼합 신호가 나타납니다. {summary}

시장 구조에서 명확한 방향성이 아직 확정되지 않았습니다.
FVG(공정가치갭)가 존재한다면, 해당 구간까지의 되돌림 후 
반응을 확인하는 것이 좋습니다."""
    
    else:
        return f"""ICT 분석 결과, 하방 유동성 확보 가능성이 있습니다. {summary}

현재 가격 아래에 유동성(Liquidity Pool)이 밀집되어 있어,
스마트 머니가 이 유동성을 먼저 확보(Liquidity Grab)한 후 
반등할 가능성을 염두에 둬야 합니다. 섣부른 진입보다는 
유동성 회수 후 반등 확인 시 진입하는 것이 안전합니다."""


def _generate_wyckoff_narrative(report: ComprehensiveReport) -> str:
    """Generate Wyckoff narrative"""
    score = report.wyckoff_score
    summary = report.wyckoff_summary
    
    if "Markup" in summary or "상승" in summary:
        return f"""Wyckoff 분석 결과, 현재 시장은 상승 추세(Markup) 국면에 있습니다.
{summary}

이전에 명확한 매집(Accumulation) 구간을 거쳐 레인지를 상향 돌파했습니다.
Wyckoff 이론에 따르면, Phase E(마크업 지속)에서는 조정을 매수 기회로 
활용하는 전략이 유효합니다."""
    
    elif "Accumulation" in summary or "매집" in summary:
        return f"""Wyckoff 분석 결과, 현재 시장은 매집(Accumulation) 국면입니다.
{summary}

스마트 머니가 물량을 축적하고 있는 단계로 해석됩니다.
'스프링(Spring)' 패턴이 관찰된다면, 이는 매집 완료와 상승 
전환의 강력한 신호가 됩니다. 인내심을 갖고 돌파 시 진입을 노리세요."""
    
    elif "Distribution" in summary or "분산" in summary:
        return f"""Wyckoff 분석 결과, 현재 시장은 분산(Distribution) 국면 가능성이 있습니다.
{summary}

⚠️ 스마트 머니가 고점에서 물량을 정리하는 단계일 수 있습니다.
상방 돌파 시도(UTAD)가 실패한다면 본격적인 하락 추세로 전환될 수 있으니,
롱 포지션은 보수적으로 접근하고 손절 라인을 엄격히 관리해야 합니다."""
    
    elif "Markdown" in summary or "하락" in summary:
        return f"""Wyckoff 분석 결과, 현재 시장은 하락 추세(Markdown) 국면입니다.
{summary}

⚠️ 분산(Distribution) 이후 본격적인 하락 국면으로, 
반등은 단기적이고 추세는 하방으로 유지될 가능성이 높습니다.
숏 포지션이 유리하며, 롱 진입 시 극도로 신중해야 합니다."""
    
    else:
        return f"""Wyckoff 분석 결과, 현재 시장 국면이 명확하지 않습니다.
{summary}

시장이 전환점에 있을 수 있으며, 추가적인 가격 움직임을 통해 
국면을 확인해야 합니다. Spring 또는 UTAD 같은 키 이벤트 발생 시 
방향성이 더 명확해질 것입니다."""


def _generate_executive_summary(report: ComprehensiveReport) -> str:
    """Generate executive summary"""
    return f"""HTF-LTF : {report.htf_ltf_summary}
Valuation : {report.valuation_summary}
On-Chain : {report.onchain_summary}
VPA : {report.vpa_summary}
ICT : {report.ict_summary}
Wyckoff : {report.wyckoff_summary}"""


def _get_market_condition(report: ComprehensiveReport) -> str:
    """Get market condition description"""
    if report.total_score >= 70:
        return "강세 신호가 우세한 상태로, 매수 포지션에 유리한 환경입니다"
    elif report.total_score >= 55:
        return "혼합 신호 속에서도 상승 가능성이 존재하나, 신중한 접근이 필요합니다"
    elif report.total_score >= 45:
        return "방향성이 불명확하여, 명확한 신호 출현 시까지 관망이 권장됩니다"
    elif report.total_score >= 30:
        return "약세 신호가 다소 우세하나, 반등 가능성도 열어둬야 합니다"
    else:
        return "하방 압력이 강해 매도 또는 숏 포지션이 유리한 환경입니다"


def _get_final_judgment_reason(report: ComprehensiveReport) -> str:
    """Get final judgment reasoning"""
    if report.final_bias == "매수 우위":
        factors = ", ".join(report.bullish_factors[:2]) if report.bullish_factors else "기술적 분석"
        return f"{factors} 등 추세의 근본적인 힘이 상승 방향을 지지하기 때문입니다."
    elif report.final_bias == "매도 우위":
        factors = ", ".join(report.bearish_factors[:2]) if report.bearish_factors else "기술적 분석"
        return f"{factors} 등 하방 압력이 우세하기 때문입니다."
    else:
        return "강세와 약세 신호가 혼재하여 명확한 방향성을 판단하기 어렵기 때문입니다."


def _generate_consensus_narrative(report: ComprehensiveReport) -> str:
    """Generate consensus and conflict narrative"""
    
    bullish_text = ""
    if report.bullish_factors:
        factors_text = "\n".join(f"  ✅ {f}" for f in report.bullish_factors)
        bullish_text = f"""
【합의된 강세 신호】
{factors_text}

이들 지표는 현재 시장의 주도권이 매수 세력에게 있음을 지지합니다."""
    else:
        bullish_text = "【합의된 강세 신호】\n  현재 명확한 강세 신호가 부재합니다."
    
    bearish_text = ""
    if report.bearish_factors:
        factors_text = "\n".join(f"  ⚠️ {f}" for f in report.bearish_factors)
        bearish_text = f"""
【상충되는 약세 신호】
{factors_text}

이들 지표는 단기 조정 또는 하방 위험을 경고합니다."""
    else:
        bearish_text = "【상충되는 약세 신호】\n  현재 주요 약세 신호가 부재합니다."
    
    resolution = f"""
【CIO 최종 판단】
{report.consensus}

{report.conflict_resolution if report.conflict_resolution else '모든 분석이 한 방향을 가리키고 있어 충돌이 없습니다.'}

{'다만, 신뢰도가 ' + report.conviction + ' 수준이므로 ' + ('과감한 진입이 가능합니다.' if report.conviction == 'HIGH' else '리스크 관리에 유의하며 진입해야 합니다.' if report.conviction == 'MEDIUM' else '진입 시 극도로 신중해야 하며, 포지션 사이즈를 최소화하는 것이 좋습니다.')}"""
    
    return bullish_text + "\n" + bearish_text + "\n" + resolution


def _generate_trading_plan_narrative(report: ComprehensiveReport) -> str:
    """Generate trading plan narrative"""
    
    direction = "롱(Long)" if report.final_bias == "매수 우위" else "숏(Short)" if report.final_bias == "매도 우위" else "양방향 대기"
    
    risk_pct = abs((report.entry_price - report.stop_loss) / report.entry_price * 100) if report.entry_price > 0 else 0
    reward_pct = abs((report.take_profit_2 - report.entry_price) / report.entry_price * 100) if report.entry_price > 0 else 0
    
    return f"""
현재 포지션 권고: {report.recommended_action}
방향성(Bias): {report.final_bias}
신뢰도: {report.conviction}

【구체적 실행 계획】

📍 진입 전략:
   방향: {direction}
   최적 진입가: ${report.entry_price:,.4f}
   진입 구간: ${report.entry_zone[0]:,.4f} ~ ${report.entry_zone[1]:,.4f}
   
🎯 익절 목표:
   1차 TP (1:1 R:R): ${report.take_profit_1:,.4f}
   2차 TP (1:2 R:R): ${report.take_profit_2:,.4f} (+{reward_pct:.1f}%)
   3차 TP (1:3 R:R): ${report.take_profit_3:,.4f}

🛑 손절 기준:
   손절가: ${report.stop_loss:,.4f} (-{risk_pct:.1f}%)
   
📊 리스크/리워드:
   R:R 비율: 1:{report.risk_reward_ratio:.1f}
   
💡 레버리지 권장:
   {report.leverage_recommendation}
   {'(신뢰도가 높아 다소 공격적인 진입이 가능합니다)' if report.conviction == 'HIGH' else '(불확실성이 있으므로 보수적인 레버리지 운용이 필수적입니다)' if report.conviction == 'LOW' else '(적절한 리스크 관리 하에 진입하세요)'}
   
⚠️ 주의사항:
   - 진입 전 반드시 현재가와 제시된 진입가를 비교하세요
   - 손절 라인은 절대적으로 준수해야 합니다
   - 분할 진입/익절로 리스크를 분산하는 것을 권장합니다"""


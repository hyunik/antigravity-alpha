"""
CIO (Chief Investment Officer) Agent
LLM-powered reasoning agent for final trade decision making
Supports both OpenAI and Gemini
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from loguru import logger

from config.settings import settings
from analysis.scoring_engine import CoinScore


@dataclass
class TradeRecommendation:
    """Final trade recommendation from CIO Agent"""
    symbol: str
    direction: str  # "LONG", "SHORT"
    conviction: str  # "HIGH", "MEDIUM", "LOW"
    score: float
    
    # Price levels
    current_price: float
    entry_zone: tuple
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    
    # Risk/Reward
    risk_percent: float
    reward_1_percent: float
    reward_2_percent: float
    reward_3_percent: float
    
    # Analysis
    key_reasons: List[str]
    risk_factors: List[str]
    agent_reasoning: str
    
    # Metadata
    timestamp: str
    timeframe: str = "4h"


class CIOAgent:
    """
    CIO (Chief Investment Officer) Agent
    
    Uses LLM to:
    1. Validate technical/on-chain confluence
    2. Generate detailed reasoning
    3. Refine entry/exit levels
    4. Produce final trade recommendations
    """
    
    SYSTEM_PROMPT = """당신은 암호화폐 트레이딩 펌의 최고투자책임자(CIO)입니다.
기술적 분석과 시장 데이터를 분석하여 최종 매매 결정을 내리는 역할을 합니다.

다음 4단계 분석 프로세스를 따릅니다:
1. **구조 분석**: 현재 시장 추세와 Wyckoff 국면 파악
2. **셋업 식별**: ICT (MSS, FVG)와 VCP 패턴 간의 컨플루언스 확인
3. **데이터 검증**: 기술적 셋업과 시장 지표 간의 정합성 확인
4. **리스크 평가**: 스퀴즈 위험 평가 및 포지션 사이징 결정

점수 가이드라인:
- 80점 이상: 모든 지표 정렬 - 공격적 진입 권장
- 60-79점: 좋은 기술적 셋업이나 일부 데이터 부족 - 분할 매수 접근
- 60점 미만: 컨플루언스 부족 - 이 거래는 패스

**과거 성과 기반 학습**:
{performance_insights}

이 인사이트를 적용하여 판단 정확도를 개선하세요.

항상 제공해야 할 것:
- 명확한 방향 (LONG/SHORT)
- 진입 구간과 그 근거
- 패턴 무효화 기준 손절가
- R:R 비율에 따른 다중 목표가
- 모니터링할 주요 리스크

**중요: 모든 분석과 설명은 반드시 한국어로 작성하세요.**"""

    ANALYSIS_PROMPT = """Analyze this cryptocurrency trading opportunity:

## Coin: {symbol}
## Current Price: ${current_price:.4f}

## Technical Analysis Scores:
- ICT Score: {ict_score:.1f}/100 (MSS: {has_mss}, FVG: {has_fvg})
- Wyckoff Score: {wyckoff_score:.1f}/100 (Phase: {wyckoff_phase}, Spring: {has_spring})
- VCP Score: {vcp_score:.1f}/100 (VCP Pattern: {has_vcp})
- Market Score: {market_score:.1f}/100

## Total Score: {total_score:.1f}/100
## Suggested Direction: {direction}

## Key Patterns Detected:
{reasons}

## Potential Risk Factors:
{risk_factors}

## Suggested Trade Setup:
- Entry Zone: ${entry_low:.4f} - ${entry_high:.4f}
- Stop Loss: ${stop_loss:.4f}
- Target 1 (1:1 R:R): ${target_1:.4f}
- Target 2 (1:2 R:R): ${target_2:.4f}
- Target 3 (1:3 R:R): ${target_3:.4f}

Please provide your analysis following the 4-step reasoning process. Be specific about:
1. Why this setup is valid or should be skipped
2. Any adjustments to entry/exit levels
3. Position sizing recommendation (aggressive/scale-in/pass)
4. Key levels to watch

**IMPORTANT: Write ALL your reasoning and analysis in Korean (한국어로 작성).**

Format your response as JSON with the following structure:
{{
    "decision": "ENTER" or "PASS",
    "reasoning": "한국어로 상세한 4단계 분석 작성",
    "adjusted_entry_low": optional number,
    "adjusted_entry_high": optional number,
    "adjusted_stop_loss": optional number,
    "position_sizing": "AGGRESSIVE" or "SCALE_IN" or "CONSERVATIVE",
    "key_levels_to_watch": ["레벨1", "레벨2"],
    "additional_risks": ["리스크1", "리스크2"]
}}"""

    def __init__(self):
        self.provider = settings.llm.provider
        self._client = None
        self._performance_insights = self._load_performance_insights()
    
    def _load_performance_insights(self) -> str:
        """Load performance insights from historical data for LLM learning"""
        try:
            from analysis.performance_tracker import PerformanceTracker
            
            tracker = PerformanceTracker()
            tracker.load_recommendations()
            
            if len(tracker.recommendations) < 5:
                return "- Not enough historical data yet. Use standard judgment."
            
            stats = tracker.calculate_stats(days=30)
            
            insights = []
            
            # Overall performance
            if stats.win_rate > 0:
                insights.append(f"- Overall win rate: {stats.win_rate:.1f}% (higher score = better accuracy)")
            
            # Conviction-based insights
            if stats.by_conviction:
                for conv, data in stats.by_conviction.items():
                    if data.get("win_rate", 0) >= 70:
                        insights.append(f"- {conv} conviction signals have {data['win_rate']:.0f}% win rate - RELIABLE")
                    elif data.get("win_rate", 0) < 50:
                        insights.append(f"- {conv} conviction signals underperform ({data['win_rate']:.0f}%) - BE MORE SELECTIVE")
            
            # Direction-based insights
            if stats.by_direction:
                for direction, data in stats.by_direction.items():
                    insights.append(f"- {direction} trades: {data['win_rate']:.0f}% win rate, avg P&L {data['avg_pnl']:+.1f}%")
            
            # Pattern effectiveness
            if stats.hit_tp1_count > 0:
                tp1_rate = (stats.hit_tp1_count / stats.total_recommendations) * 100
                insights.append(f"- TP1 hit rate: {tp1_rate:.0f}% - Targets are {'realistic' if tp1_rate > 50 else 'too aggressive'}")
            
            # SL analysis
            if stats.hit_sl_count > 0:
                sl_rate = (stats.hit_sl_count / stats.total_recommendations) * 100
                if sl_rate > 30:
                    insights.append(f"- SL hit rate {sl_rate:.0f}% is high - Consider wider stops or better entry timing")
            
            return "\n".join(insights) if insights else "- Standard performance. Apply normal judgment."
            
        except Exception as e:
            logger.debug(f"Could not load performance insights: {e}")
            return "- Performance data unavailable. Use standard judgment."
    
    def _get_system_prompt(self) -> str:
        """Get system prompt with performance insights"""
        return self.SYSTEM_PROMPT.format(performance_insights=self._performance_insights)
    
    def _get_client(self):
        """Get or initialize LLM client"""
        if self._client is None:
            self._client = settings.get_llm_client()
        return self._client
    
    async def analyze_coin(self, coin_score: CoinScore, current_price: float) -> tuple:
        """
        Analyze a coin using LLM reasoning
        
        Args:
            coin_score: CoinScore from scoring engine
            current_price: Current market price
            
        Returns:
            Tuple of (TradeRecommendation or None, pass_reason or None)
        """
        # Skip low-conviction setups to save API calls
        if coin_score.total_score < 50:
            logger.debug(f"Skipping {coin_score.symbol} - score too low ({coin_score.total_score:.1f})")
            return (None, "점수가 너무 낮음 (50점 미만)")
        
        # Format prompt
        prompt = self.ANALYSIS_PROMPT.format(
            symbol=coin_score.symbol,
            current_price=current_price,
            ict_score=coin_score.ict_score,
            has_mss=coin_score.has_mss,
            has_fvg=coin_score.has_fvg,
            wyckoff_score=coin_score.wyckoff_score,
            wyckoff_phase=coin_score.wyckoff_phase,
            has_spring=coin_score.has_spring,
            vcp_score=coin_score.vcp_score,
            has_vcp=coin_score.has_vcp,
            market_score=coin_score.market_score,
            total_score=coin_score.total_score,
            direction=coin_score.direction,
            reasons="\n".join(f"- {r}" for r in coin_score.reasons) or "- No strong patterns",
            risk_factors="\n".join(f"- {r}" for r in coin_score.risk_factors) or "- No significant risks",
            entry_low=coin_score.entry_zone[0],
            entry_high=coin_score.entry_zone[1],
            stop_loss=coin_score.stop_loss,
            target_1=coin_score.targets[0] if len(coin_score.targets) > 0 else current_price * 1.05,
            target_2=coin_score.targets[1] if len(coin_score.targets) > 1 else current_price * 1.10,
            target_3=coin_score.targets[2] if len(coin_score.targets) > 2 else current_price * 1.15,
        )
        
        try:
            # Call LLM
            response = await self._call_llm(prompt)
            
            if not response:
                return (self._create_fallback_recommendation(coin_score, current_price), None)
            
            # Parse response
            parsed = self._parse_response(response)
            
            if parsed.get("decision") == "PASS":
                pass_reason = parsed.get('reasoning', '분석 결과 진입 부적합')
                logger.info(f"CIO Agent passed on {coin_score.symbol}: {pass_reason[:100]}")
                return (None, pass_reason)
            
            # Create recommendation with potential adjustments
            return (self._create_recommendation(coin_score, current_price, parsed), None)
            
        except Exception as e:
            logger.error(f"CIO Agent analysis failed for {coin_score.symbol}: {e}")
            return (self._create_fallback_recommendation(coin_score, current_price), None)
    
    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API based on provider"""
        try:
            if self.provider == "openai":
                return await self._call_openai(prompt)
            elif self.provider == "gemini":
                return await self._call_gemini(prompt)
            else:
                logger.error(f"Unknown LLM provider: {self.provider}")
                return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    async def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        import asyncio
        
        client = self._get_client()
        
        def sync_call():
            response = client.chat.completions.create(
                model=settings.llm.openai_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        
        # Run sync call in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_call)
    
    async def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini API"""
        import asyncio
        
        client = self._get_client()
        
        def sync_call():
            full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"
            response = client.generate_content(full_prompt)
            return response.text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_call)
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response, extracting JSON if present"""
        try:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            return {"decision": "ENTER", "reasoning": response}
        except json.JSONDecodeError:
            return {"decision": "ENTER", "reasoning": response}
    
    def _create_recommendation(
        self,
        coin_score: CoinScore,
        current_price: float,
        llm_response: Dict
    ) -> TradeRecommendation:
        """Create trade recommendation from analysis"""
        from datetime import datetime
        
        # Use adjusted levels if provided
        entry_low = llm_response.get("adjusted_entry_low", coin_score.entry_zone[0])
        entry_high = llm_response.get("adjusted_entry_high", coin_score.entry_zone[1])
        stop_loss = llm_response.get("adjusted_stop_loss", coin_score.stop_loss)
        
        # Calculate risk/reward percentages
        entry_mid = (entry_low + entry_high) / 2
        risk_percent = abs((entry_mid - stop_loss) / entry_mid) * 100
        
        targets = coin_score.targets
        reward_1 = abs((targets[0] - entry_mid) / entry_mid) * 100 if len(targets) > 0 else 0
        reward_2 = abs((targets[1] - entry_mid) / entry_mid) * 100 if len(targets) > 1 else 0
        reward_3 = abs((targets[2] - entry_mid) / entry_mid) * 100 if len(targets) > 2 else 0
        
        # Combine reasons with LLM insights
        key_reasons = coin_score.reasons.copy()
        if llm_response.get("key_levels_to_watch"):
            key_reasons.append(f"Key levels: {', '.join(llm_response['key_levels_to_watch'])}")
        
        risk_factors = coin_score.risk_factors.copy()
        if llm_response.get("additional_risks"):
            risk_factors.extend(llm_response["additional_risks"])
        
        return TradeRecommendation(
            symbol=coin_score.symbol,
            direction=coin_score.direction,
            conviction=coin_score.conviction,
            score=coin_score.total_score,
            current_price=current_price,
            entry_zone=(entry_low, entry_high),
            stop_loss=stop_loss,
            target_1=targets[0] if len(targets) > 0 else current_price * 1.05,
            target_2=targets[1] if len(targets) > 1 else current_price * 1.10,
            target_3=targets[2] if len(targets) > 2 else current_price * 1.15,
            risk_percent=risk_percent,
            reward_1_percent=reward_1,
            reward_2_percent=reward_2,
            reward_3_percent=reward_3,
            key_reasons=key_reasons,
            risk_factors=risk_factors,
            agent_reasoning=llm_response.get("reasoning", ""),
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _create_fallback_recommendation(
        self,
        coin_score: CoinScore,
        current_price: float
    ) -> TradeRecommendation:
        """Create recommendation without LLM (fallback)"""
        return self._create_recommendation(
            coin_score,
            current_price,
            {
                "decision": "ENTER",
                "reasoning": "Automated analysis based on technical indicators (LLM unavailable)"
            }
        )
    
    async def batch_analyze(
        self,
        coin_scores: List[CoinScore],
        price_map: Dict[str, float],
        min_score: float = 60,
        max_recommendations: int = 50
    ) -> tuple:
        """
        Analyze multiple coins and return top recommendations
        
        Args:
            coin_scores: List of CoinScore from scoring engine
            price_map: Dict mapping symbol to current price
            min_score: Minimum score to analyze
            max_recommendations: Maximum recommendations to return
            
        Returns:
            Tuple of (List[TradeRecommendation], Dict[symbol -> pass_reason])
        """
        # Filter and sort by score
        qualified = sorted(
            [s for s in coin_scores if s.total_score >= min_score],
            key=lambda x: x.total_score,
            reverse=True
        )[:max_recommendations]
        
        recommendations = []
        pass_reasons = {}  # Track why coins were passed
        
        for score in qualified:
            price = price_map.get(score.symbol, 0)
            if price > 0:
                rec, reason = await self.analyze_coin(score, price)
                if rec:
                    recommendations.append(rec)
                elif reason:
                    pass_reasons[score.symbol] = {
                        "score": score.total_score,
                        "direction": score.direction,
                        "reason": reason
                    }
        
        sorted_recs = sorted(recommendations, key=lambda x: x.score, reverse=True)
        return (sorted_recs, pass_reasons)


async def test_cio_agent():
    """Test CIO Agent"""
    from analysis.scoring_engine import CoinScore
    
    # Create a mock CoinScore
    mock_score = CoinScore(
        symbol="BTCUSDT",
        total_score=75.5,
        ict_score=80,
        wyckoff_score=70,
        vcp_score=65,
        market_score=85,
        conviction="MEDIUM",
        direction="LONG",
        has_mss=True,
        has_fvg=True,
        has_spring=False,
        has_vcp=True,
        wyckoff_phase="Accumulation",
        entry_zone=(42000, 43000),
        stop_loss=41000,
        targets=[44000, 46000, 50000],
        reasons=["MSS detected", "4h FVG", "VCP tightness"],
        risk_factors=["High funding rate"]
    )
    
    agent = CIOAgent()
    
    # Note: This will fail without API keys
    print(f"Testing CIO Agent with provider: {agent.provider}")
    print(f"Mock score for BTCUSDT: {mock_score.total_score}")
    print("Note: Actual LLM call requires valid API keys in .env")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_cio_agent())

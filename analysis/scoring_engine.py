"""
Scoring Engine
Combines all analysis results into a unified scoring system
"""

import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
from loguru import logger

from .ict_detector import ICTDetector
from .wyckoff_detector import WyckoffDetector, WyckoffPhase
from .vcp_detector import VCPDetector


@dataclass
class CoinScore:
    """Unified coin analysis score"""
    symbol: str
    total_score: float  # 0-100
    ict_score: float
    wyckoff_score: float
    vcp_score: float
    market_score: float  # Based on OI, funding, volume
    
    # Conviction level
    conviction: str  # "HIGH", "MEDIUM", "LOW"
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    
    # Key patterns detected
    has_mss: bool
    has_fvg: bool
    has_spring: bool
    has_vcp: bool
    wyckoff_phase: str
    
    # Trade setup
    entry_zone: tuple  # (low, high)
    stop_loss: float
    targets: List[float]  # [TP1, TP2, TP3]
    
    # Key reasons
    reasons: List[str]
    risk_factors: List[str]


class ScoringEngine:
    """
    Unified Scoring Engine
    
    Combines ICT, Wyckoff, and VCP analysis with market data
    to produce a final score and trade recommendation
    """
    
    # Score weights
    WEIGHTS = {
        "ict": 0.30,
        "wyckoff": 0.25,
        "vcp": 0.20,
        "market": 0.25
    }
    
    def __init__(self):
        self.ict_detector = ICTDetector()
        self.wyckoff_detector = WyckoffDetector()
        self.vcp_detector = VCPDetector()
    
    def analyze_market_data(
        self,
        open_interest: float,
        funding_rate: float,
        df: pd.DataFrame
    ) -> Dict:
        """
        Analyze market data (OI, Funding Rate, Volume)
        
        Args:
            open_interest: Current open interest
            funding_rate: Current funding rate
            df: OHLCV DataFrame for volume analysis
            
        Returns:
            Dict with market analysis results
        """
        score = 50  # Start neutral
        signals = []
        
        # Funding rate analysis
        if funding_rate > 0.01:
            # High positive funding = potential long squeeze
            score -= 15
            signals.append("High funding rate (potential long squeeze)")
        elif funding_rate < -0.01:
            # Negative funding = potential short squeeze
            score += 15
            signals.append("Negative funding rate (potential short squeeze)")
        elif abs(funding_rate) < 0.005:
            score += 5
            signals.append("Neutral funding rate")
        
        # Volume analysis
        if len(df) >= 20:
            recent_vol = df["volume"].tail(5).mean()
            avg_vol = df["volume"].iloc[-20:-5].mean()
            
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                if vol_ratio > 1.5:
                    score += 10
                    signals.append("Volume surge detected")
                elif vol_ratio < 0.5:
                    score -= 5
                    signals.append("Low volume environment")
        
        # Price momentum
        if len(df) >= 5:
            price_change = (df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]
            if price_change > 0.05:
                score += 10
                signals.append("Strong bullish momentum")
            elif price_change < -0.05:
                score -= 10
                signals.append("Strong bearish momentum")
        
        return {
            "market_score": min(max(score, 0), 100),
            "signals": signals,
            "funding_rate": funding_rate,
            "is_bullish_funding": funding_rate < 0,
            "is_bearish_funding": funding_rate > 0.01
        }
    
    def calculate_entry_zones(
        self,
        df: pd.DataFrame,
        direction: str,
        ict_analysis: Dict,
        wyckoff_analysis,
        vcp_analysis: Dict
    ) -> Dict:
        """
        Calculate entry zone, stop loss, and targets
        
        Args:
            df: OHLCV DataFrame
            direction: "LONG" or "SHORT"
            ict_analysis: ICT analysis results
            wyckoff_analysis: Wyckoff analysis results
            vcp_analysis: VCP analysis results
            
        Returns:
            Dict with entry, SL, and targets
        """
        current_price = df["close"].iloc[-1]
        atr = (df["high"] - df["low"]).tail(14).mean()
        
        if direction == "LONG":
            # Entry zone near support/FVG
            if wyckoff_analysis.support_level:
                entry_low = wyckoff_analysis.support_level
            else:
                entry_low = current_price - atr
            
            entry_high = current_price
            
            # Stop loss below support
            stop_loss = entry_low - (atr * 0.5)
            
            # Targets based on risk
            risk = entry_high - stop_loss
            targets = [
                entry_high + risk * 1.0,  # 1:1 R:R
                entry_high + risk * 2.0,  # 1:2 R:R
                entry_high + risk * 3.0,  # 1:3 R:R
            ]
            
            # Adjust target to resistance if available
            if wyckoff_analysis.resistance_level:
                targets[1] = max(targets[1], wyckoff_analysis.resistance_level)
        
        else:  # SHORT
            entry_low = current_price
            if wyckoff_analysis.resistance_level:
                entry_high = wyckoff_analysis.resistance_level
            else:
                entry_high = current_price + atr
            
            stop_loss = entry_high + (atr * 0.5)
            
            risk = stop_loss - entry_low
            targets = [
                entry_low - risk * 1.0,
                entry_low - risk * 2.0,
                entry_low - risk * 3.0,
            ]
        
        return {
            "entry_zone": (entry_low, entry_high),
            "stop_loss": stop_loss,
            "targets": targets
        }
    
    def score_coin(
        self,
        symbol: str,
        df_4h: pd.DataFrame,
        df_1d: Optional[pd.DataFrame] = None,
        open_interest: float = 0,
        funding_rate: float = 0
    ) -> CoinScore:
        """
        Perform comprehensive analysis and scoring for a coin
        
        Args:
            symbol: Trading pair symbol
            df_4h: 4-hour OHLCV DataFrame
            df_1d: Daily OHLCV DataFrame (optional)
            open_interest: Current open interest
            funding_rate: Current funding rate
            
        Returns:
            CoinScore with full analysis
        """
        # Use daily for Wyckoff phase, 4h for patterns
        analysis_df = df_1d if df_1d is not None and len(df_1d) >= 50 else df_4h
        
        # Run all detectors
        ict_result = self.ict_detector.analyze(df_4h)
        wyckoff_result = self.wyckoff_detector.analyze(analysis_df)
        vcp_result = self.vcp_detector.analyze(df_4h)
        market_result = self.analyze_market_data(open_interest, funding_rate, df_4h)
        
        # Extract scores
        ict_score = ict_result["ict_score"]
        wyckoff_score = wyckoff_result.wyckoff_score
        vcp_score = vcp_result["vcp_score"]
        market_score = market_result["market_score"]
        
        # Calculate weighted total score
        total_score = (
            ict_score * self.WEIGHTS["ict"] +
            wyckoff_score * self.WEIGHTS["wyckoff"] +
            vcp_score * self.WEIGHTS["vcp"] +
            market_score * self.WEIGHTS["market"]
        )
        
        # Determine direction
        bullish_signals = 0
        bearish_signals = 0
        
        if ict_result["bias"] == "bullish":
            bullish_signals += 1
        elif ict_result["bias"] == "bearish":
            bearish_signals += 1
        
        if wyckoff_result.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP]:
            bullish_signals += 1
        elif wyckoff_result.phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]:
            bearish_signals += 1
        
        if market_result["is_bullish_funding"]:
            bullish_signals += 1
        elif market_result["is_bearish_funding"]:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            direction = "LONG"
        elif bearish_signals > bullish_signals:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        
        # Determine conviction
        if total_score >= 80:
            conviction = "HIGH"
        elif total_score >= 60:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"
        
        # Calculate trade setup
        trade_setup = self.calculate_entry_zones(
            df_4h, direction, ict_result, wyckoff_result, vcp_result
        )
        
        # Compile reasons
        reasons = []
        risk_factors = []
        
        if ict_result["has_mss"]:
            reasons.append(f"Market Structure Shift ({ict_result['bias']})")
        if ict_result["has_fvg"]:
            reasons.append("Fair Value Gap detected")
        if wyckoff_result.has_spring:
            reasons.append("Wyckoff Spring pattern")
        if wyckoff_result.phase != WyckoffPhase.UNKNOWN:
            reasons.append(f"Wyckoff {wyckoff_result.phase.value} phase")
        if vcp_result["has_vcp"]:
            reasons.append(f"VCP pattern (strength: {vcp_result['vcp_pattern'].strength:.2f})")
        if vcp_result["is_tight"]:
            reasons.append("Volatility contraction (tight range)")
        if vcp_result["has_volume_dryup"]:
            reasons.append("Volume dry-up detected")
        
        for signal in market_result["signals"]:
            if "squeeze" in signal.lower() or "surge" in signal.lower():
                reasons.append(signal)
        
        # Risk factors
        if funding_rate > 0.01:
            risk_factors.append("High positive funding - potential long squeeze risk")
        if funding_rate < -0.01:
            risk_factors.append("Negative funding - potential short squeeze risk")
        if wyckoff_result.phase == WyckoffPhase.DISTRIBUTION:
            risk_factors.append("Distribution phase - potential reversal")
        
        return CoinScore(
            symbol=symbol,
            total_score=total_score,
            ict_score=ict_score,
            wyckoff_score=wyckoff_score,
            vcp_score=vcp_score,
            market_score=market_score,
            conviction=conviction,
            direction=direction,
            has_mss=ict_result["has_mss"],
            has_fvg=ict_result["has_fvg"],
            has_spring=wyckoff_result.has_spring,
            has_vcp=vcp_result["has_vcp"],
            wyckoff_phase=wyckoff_result.phase.value,
            entry_zone=trade_setup["entry_zone"],
            stop_loss=trade_setup["stop_loss"],
            targets=trade_setup["targets"],
            reasons=reasons,
            risk_factors=risk_factors
        )
    
    def rank_coins(self, scores: List[CoinScore], min_score: float = 60) -> List[CoinScore]:
        """
        Rank and filter coins by score
        
        Args:
            scores: List of CoinScore objects
            min_score: Minimum score to include
            
        Returns:
            Sorted list of qualifying coins
        """
        qualified = [s for s in scores if s.total_score >= min_score]
        return sorted(qualified, key=lambda x: x.total_score, reverse=True)


def test_scoring_engine():
    """Test scoring engine"""
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=100, freq="4H")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices + np.random.randn(100) * 0.2,
        "high": prices + np.abs(np.random.randn(100) * 1),
        "low": prices - np.abs(np.random.randn(100) * 1),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 100)
    })
    
    engine = ScoringEngine()
    score = engine.score_coin(
        symbol="BTCUSDT",
        df_4h=df,
        open_interest=1000000,
        funding_rate=0.005
    )
    
    print(f"Symbol: {score.symbol}")
    print(f"Total Score: {score.total_score:.1f}")
    print(f"  ICT: {score.ict_score:.1f}")
    print(f"  Wyckoff: {score.wyckoff_score:.1f}")
    print(f"  VCP: {score.vcp_score:.1f}")
    print(f"  Market: {score.market_score:.1f}")
    print(f"Direction: {score.direction}")
    print(f"Conviction: {score.conviction}")
    print(f"Entry Zone: ${score.entry_zone[0]:.2f} - ${score.entry_zone[1]:.2f}")
    print(f"Stop Loss: ${score.stop_loss:.2f}")
    print(f"Targets: {[f'${t:.2f}' for t in score.targets]}")
    print(f"Reasons: {score.reasons}")


if __name__ == "__main__":
    test_scoring_engine()

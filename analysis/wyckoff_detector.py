"""
Wyckoff Phase Detector
Detects Wyckoff market phases and Spring patterns
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class WyckoffPhase(Enum):
    """Wyckoff Market Phases"""
    ACCUMULATION = "Accumulation"
    MARKUP = "Markup"
    DISTRIBUTION = "Distribution"
    MARKDOWN = "Markdown"
    UNKNOWN = "Unknown"


@dataclass
class Spring:
    """Wyckoff Spring detection result"""
    timestamp: pd.Timestamp
    spring_low: float
    support_level: float
    recovery_price: float
    volume_surge: float  # Volume ratio vs average
    strength: float  # 0-1


@dataclass
class WyckoffAnalysis:
    """Wyckoff analysis result"""
    phase: WyckoffPhase
    phase_confidence: float
    has_spring: bool
    springs: List[Spring]
    support_level: Optional[float]
    resistance_level: Optional[float]
    wyckoff_score: float


class WyckoffDetector:
    """
    Wyckoff Method Pattern Detector
    
    Detects:
    - Current market phase (Accumulation, Markup, Distribution, Markdown)
    - Spring patterns (Phase C shakeout)
    - Support/Resistance levels
    """
    
    def __init__(
        self,
        lookback_period: int = 50,
        support_resistance_lookback: int = 20,
        volume_surge_threshold: float = 1.5
    ):
        """
        Initialize Wyckoff Detector
        
        Args:
            lookback_period: Period for phase detection
            support_resistance_lookback: Period for S/R detection
            volume_surge_threshold: Volume multiplier for surge detection
        """
        self.lookback_period = lookback_period
        self.sr_lookback = support_resistance_lookback
        self.volume_surge_threshold = volume_surge_threshold
    
    def detect_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Detect key support and resistance levels
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        if len(df) < self.sr_lookback:
            return df["low"].min(), df["high"].max()
        
        recent = df.tail(self.sr_lookback)
        
        # Find swing lows for support
        lows = recent["low"].values
        swing_lows = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        support = np.mean(swing_lows) if swing_lows else recent["low"].min()
        
        # Find swing highs for resistance
        highs = recent["high"].values
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
        
        resistance = np.mean(swing_highs) if swing_highs else recent["high"].max()
        
        return support, resistance
    
    def detect_phase(self, df: pd.DataFrame) -> Tuple[WyckoffPhase, float]:
        """
        Detect current Wyckoff phase
        
        Uses volatility, trend, and volume characteristics to determine phase
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (phase, confidence)
        """
        if len(df) < self.lookback_period:
            return WyckoffPhase.UNKNOWN, 0.0
        
        recent = df.tail(self.lookback_period)
        
        # Calculate metrics
        price_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]
        volatility = recent["close"].pct_change().std()
        
        # Volume trend
        first_half_vol = recent["volume"].iloc[:len(recent)//2].mean()
        second_half_vol = recent["volume"].iloc[len(recent)//2:].mean()
        volume_trend = (second_half_vol - first_half_vol) / first_half_vol if first_half_vol > 0 else 0
        
        # Price range compression
        first_half_range = (recent["high"].iloc[:len(recent)//2].max() - 
                          recent["low"].iloc[:len(recent)//2].min())
        second_half_range = (recent["high"].iloc[len(recent)//2:].max() - 
                            recent["low"].iloc[len(recent)//2:].min())
        range_ratio = second_half_range / first_half_range if first_half_range > 0 else 1
        
        # Determine phase based on characteristics
        phase = WyckoffPhase.UNKNOWN
        confidence = 0.5
        
        # Accumulation: Low volatility, sideways price, volume pickup
        if abs(price_change) < 0.05 and range_ratio < 0.8 and volume_trend > 0:
            phase = WyckoffPhase.ACCUMULATION
            confidence = 0.6 + min(0.3, volume_trend * 0.3)
        
        # Markup: Strong uptrend, increasing volume
        elif price_change > 0.1 and volume_trend > 0:
            phase = WyckoffPhase.MARKUP
            confidence = 0.6 + min(0.3, price_change)
        
        # Distribution: High volatility at highs, volume divergence
        elif abs(price_change) < 0.05 and volatility > 0.02:
            # Check if near recent highs
            current_price = recent["close"].iloc[-1]
            period_high = recent["high"].max()
            if current_price > period_high * 0.95:  # Within 5% of high
                phase = WyckoffPhase.DISTRIBUTION
                confidence = 0.6 + min(0.25, volatility * 10)
        
        # Markdown: Strong downtrend
        elif price_change < -0.1:
            phase = WyckoffPhase.MARKDOWN
            confidence = 0.6 + min(0.3, abs(price_change))
        
        return phase, min(confidence, 0.95)
    
    def detect_spring(self, df: pd.DataFrame, support_level: float) -> List[Spring]:
        """
        Detect Wyckoff Spring patterns
        
        Spring: Price breaks below support then quickly recovers with high volume
        
        Args:
            df: OHLCV DataFrame
            support_level: Key support level
            
        Returns:
            List of Spring patterns detected
        """
        if len(df) < 5:
            return []
        
        springs = []
        avg_volume = df["volume"].mean()
        
        for i in range(2, len(df) - 1):
            # Check if price broke below support
            if df["low"].iloc[i] < support_level * 0.99:  # 1% tolerance
                # Check for quick recovery (within next 2 candles)
                recovery_price = df["close"].iloc[min(i+1, len(df)-1)]
                if recovery_price > support_level:
                    # Check for volume surge
                    volume_at_spring = df["volume"].iloc[i]
                    volume_ratio = volume_at_spring / avg_volume if avg_volume > 0 else 1
                    
                    if volume_ratio >= self.volume_surge_threshold:
                        # Calculate strength
                        drop_pct = (support_level - df["low"].iloc[i]) / support_level
                        recovery_pct = (recovery_price - df["low"].iloc[i]) / df["low"].iloc[i]
                        strength = min((volume_ratio / 3) * (recovery_pct / drop_pct if drop_pct > 0 else 1), 1.0)
                        
                        springs.append(Spring(
                            timestamp=df["timestamp"].iloc[i] if "timestamp" in df.columns else df.index[i],
                            spring_low=df["low"].iloc[i],
                            support_level=support_level,
                            recovery_price=recovery_price,
                            volume_surge=volume_ratio,
                            strength=strength
                        ))
        
        return springs
    
    def analyze(self, df: pd.DataFrame) -> WyckoffAnalysis:
        """
        Perform full Wyckoff analysis
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            WyckoffAnalysis result
        """
        # Detect support/resistance
        support, resistance = self.detect_support_resistance(df)
        
        # Detect phase
        phase, phase_confidence = self.detect_phase(df)
        
        # Detect springs
        springs = self.detect_spring(df, support)
        
        # Recent springs (last 10 candles)
        recent_springs = springs[-3:] if springs else []
        recent_spring_check = any(
            s.timestamp >= df.iloc[-10]["timestamp"] if "timestamp" in df.columns else True
            for s in springs
        ) if springs else False
        
        # Calculate Wyckoff score
        wyckoff_score = self._calculate_score(phase, phase_confidence, springs, df)
        
        return WyckoffAnalysis(
            phase=phase,
            phase_confidence=phase_confidence,
            has_spring=len(recent_springs) > 0 or recent_spring_check,
            springs=springs,
            support_level=support,
            resistance_level=resistance,
            wyckoff_score=wyckoff_score
        )
    
    def _calculate_score(
        self,
        phase: WyckoffPhase,
        confidence: float,
        springs: List[Spring],
        df: pd.DataFrame
    ) -> float:
        """Calculate Wyckoff score (0-100)"""
        score = 0
        
        # Phase contribution (max 40 points)
        if phase == WyckoffPhase.ACCUMULATION:
            score += 40 * confidence  # Bullish setup
        elif phase == WyckoffPhase.MARKUP:
            score += 30 * confidence  # Already in trend
        elif phase == WyckoffPhase.DISTRIBUTION:
            score += 20 * confidence  # Potential reversal
        elif phase == WyckoffPhase.MARKDOWN:
            score += 10 * confidence  # Bearish
        
        # Spring contribution (max 40 points)
        if springs:
            best_spring = max(springs, key=lambda x: x.strength)
            score += 40 * best_spring.strength
            
            # Recency bonus
            if len(df) > 0 and springs:
                last_spring = springs[-1]
                # Check if spring is recent
                if "timestamp" in df.columns:
                    recent_candles = df.tail(10)
                    if last_spring.timestamp >= recent_candles["timestamp"].iloc[0]:
                        score += 10  # Recency bonus
        
        # Volume confirmation (max 10 points)
        if len(df) > 10:
            recent_vol = df["volume"].tail(5).mean()
            prior_vol = df["volume"].iloc[-15:-5].mean()
            if recent_vol > prior_vol * 1.2:
                score += 10
        
        return min(score, 100)


def test_wyckoff_detector():
    """Test Wyckoff detector with sample data"""
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=100, freq="4H")
    
    # Simulate accumulation pattern
    base = 100
    prices = np.concatenate([
        np.linspace(110, 100, 30),  # Initial decline
        100 + np.random.randn(40) * 2,  # Sideways accumulation
        np.linspace(100, 95, 5),  # Spring
        np.linspace(96, 110, 25),  # Markup
    ])[:100]
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices + np.random.randn(100) * 0.2,
        "high": prices + np.abs(np.random.randn(100) * 1),
        "low": prices - np.abs(np.random.randn(100) * 1),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 100)
    })
    
    detector = WyckoffDetector()
    result = detector.analyze(df)
    
    print(f"Phase: {result.phase.value} (confidence: {result.phase_confidence:.2f})")
    print(f"Has Spring: {result.has_spring}")
    print(f"Support: ${result.support_level:.2f}")
    print(f"Resistance: ${result.resistance_level:.2f}")
    print(f"Wyckoff Score: {result.wyckoff_score:.1f}")


if __name__ == "__main__":
    test_wyckoff_detector()

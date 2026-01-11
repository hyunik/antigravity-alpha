"""
ICT Pattern Detector
Detects ICT concepts: Market Structure Shift (MSS) and Fair Value Gap (FVG)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class MSS:
    """Market Structure Shift detection result"""
    timestamp: pd.Timestamp
    direction: str  # "bullish" or "bearish"
    break_price: float
    previous_swing: float
    strength: float  # 0-1 based on volume and follow-through


@dataclass
class FVG:
    """Fair Value Gap detection result"""
    timestamp: pd.Timestamp
    direction: str  # "bullish" or "bearish"
    gap_high: float
    gap_low: float
    size_percent: float
    filled: bool = False


class ICTDetector:
    """
    ICT Concepts Pattern Detector
    
    Detects:
    - Market Structure Shift (MSS): Break of previous swing high/low
    - Fair Value Gap (FVG): Imbalance between 3 consecutive candles
    """
    
    def __init__(self, swing_lookback: int = 5, fvg_min_size_percent: float = 0.1):
        """
        Initialize ICT Detector
        
        Args:
            swing_lookback: Number of candles to look back for swing detection
            fvg_min_size_percent: Minimum FVG size as percentage of price
        """
        self.swing_lookback = swing_lookback
        self.fvg_min_size_percent = fvg_min_size_percent
    
    def detect_swing_highs(self, df: pd.DataFrame, lookback: Optional[int] = None) -> pd.Series:
        """
        Detect swing highs in price data
        
        A swing high is a high that is higher than `lookback` candles on both sides
        """
        lookback = lookback or self.swing_lookback
        highs = df["high"].values
        swing_highs = pd.Series(index=df.index, dtype=float)
        
        for i in range(lookback, len(highs) - lookback):
            window = highs[i-lookback:i+lookback+1]
            if highs[i] == max(window):
                swing_highs.iloc[i] = highs[i]
        
        return swing_highs
    
    def detect_swing_lows(self, df: pd.DataFrame, lookback: Optional[int] = None) -> pd.Series:
        """
        Detect swing lows in price data
        
        A swing low is a low that is lower than `lookback` candles on both sides
        """
        lookback = lookback or self.swing_lookback
        lows = df["low"].values
        swing_lows = pd.Series(index=df.index, dtype=float)
        
        for i in range(lookback, len(lows) - lookback):
            window = lows[i-lookback:i+lookback+1]
            if lows[i] == min(window):
                swing_lows.iloc[i] = lows[i]
        
        return swing_lows
    
    def detect_mss(self, df: pd.DataFrame) -> List[MSS]:
        """
        Detect Market Structure Shift patterns
        
        MSS occurs when:
        - Bullish: Price breaks above the previous significant swing high
        - Bearish: Price breaks below the previous significant swing low
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of MSS patterns detected
        """
        if len(df) < self.swing_lookback * 2 + 1:
            return []
        
        swing_highs = self.detect_swing_highs(df)
        swing_lows = self.detect_swing_lows(df)
        
        mss_patterns = []
        last_swing_high = None
        last_swing_low = None
        
        for i in range(self.swing_lookback, len(df)):
            # Update last known swing points
            if pd.notna(swing_highs.iloc[i]):
                last_swing_high = swing_highs.iloc[i]
            if pd.notna(swing_lows.iloc[i]):
                last_swing_low = swing_lows.iloc[i]
            
            # Check for bullish MSS (break of swing high)
            if last_swing_high is not None:
                if df["close"].iloc[i] > last_swing_high and df["close"].iloc[i-1] <= last_swing_high:
                    # Calculate strength based on volume spike
                    avg_volume = df["volume"].iloc[max(0, i-20):i].mean()
                    volume_ratio = df["volume"].iloc[i] / avg_volume if avg_volume > 0 else 1
                    strength = min(volume_ratio / 2, 1.0)  # Normalize to 0-1
                    
                    mss_patterns.append(MSS(
                        timestamp=df["timestamp"].iloc[i] if "timestamp" in df.columns else df.index[i],
                        direction="bullish",
                        break_price=df["close"].iloc[i],
                        previous_swing=last_swing_high,
                        strength=strength
                    ))
                    last_swing_high = None  # Reset after break
            
            # Check for bearish MSS (break of swing low)
            if last_swing_low is not None:
                if df["close"].iloc[i] < last_swing_low and df["close"].iloc[i-1] >= last_swing_low:
                    avg_volume = df["volume"].iloc[max(0, i-20):i].mean()
                    volume_ratio = df["volume"].iloc[i] / avg_volume if avg_volume > 0 else 1
                    strength = min(volume_ratio / 2, 1.0)
                    
                    mss_patterns.append(MSS(
                        timestamp=df["timestamp"].iloc[i] if "timestamp" in df.columns else df.index[i],
                        direction="bearish",
                        break_price=df["close"].iloc[i],
                        previous_swing=last_swing_low,
                        strength=strength
                    ))
                    last_swing_low = None
        
        return mss_patterns
    
    def detect_fvg(self, df: pd.DataFrame) -> List[FVG]:
        """
        Detect Fair Value Gap patterns
        
        FVG occurs when there's a gap between:
        - Bullish FVG: Candle 1's high and Candle 3's low (when Candle 2 is bullish)
        - Bearish FVG: Candle 1's low and Candle 3's high (when Candle 2 is bearish)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of FVG patterns detected
        """
        if len(df) < 3:
            return []
        
        fvg_patterns = []
        
        for i in range(2, len(df)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Bullish FVG: Gap between candle1 high and candle3 low
            if candle3["low"] > candle1["high"]:
                gap_size = candle3["low"] - candle1["high"]
                gap_percent = (gap_size / candle2["close"]) * 100
                
                if gap_percent >= self.fvg_min_size_percent:
                    fvg_patterns.append(FVG(
                        timestamp=df["timestamp"].iloc[i] if "timestamp" in df.columns else df.index[i],
                        direction="bullish",
                        gap_high=candle3["low"],
                        gap_low=candle1["high"],
                        size_percent=gap_percent
                    ))
            
            # Bearish FVG: Gap between candle1 low and candle3 high
            if candle3["high"] < candle1["low"]:
                gap_size = candle1["low"] - candle3["high"]
                gap_percent = (gap_size / candle2["close"]) * 100
                
                if gap_percent >= self.fvg_min_size_percent:
                    fvg_patterns.append(FVG(
                        timestamp=df["timestamp"].iloc[i] if "timestamp" in df.columns else df.index[i],
                        direction="bearish",
                        gap_high=candle1["low"],
                        gap_low=candle3["high"],
                        size_percent=gap_percent
                    ))
        
        return fvg_patterns
    
    def check_fvg_filled(self, fvg: FVG, df: pd.DataFrame) -> bool:
        """Check if an FVG has been filled by subsequent price action"""
        for i in range(len(df)):
            if fvg.direction == "bullish":
                if df["low"].iloc[i] <= fvg.gap_low:
                    return True
            else:
                if df["high"].iloc[i] >= fvg.gap_high:
                    return True
        return False
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform full ICT analysis on OHLCV data
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dict with analysis results
        """
        mss_patterns = self.detect_mss(df)
        fvg_patterns = self.detect_fvg(df)
        
        # Get recent patterns (last 20 candles)
        recent_idx = len(df) - 20 if len(df) > 20 else 0
        
        if "timestamp" in df.columns:
            recent_ts = df.iloc[recent_idx]["timestamp"]
            recent_mss = [m for m in mss_patterns if m.timestamp >= recent_ts]
            recent_fvg = [f for f in fvg_patterns if f.timestamp >= recent_ts]
        else:
            recent_mss = mss_patterns[-10:] if len(mss_patterns) > 10 else mss_patterns
            recent_fvg = fvg_patterns[-10:] if len(fvg_patterns) > 10 else fvg_patterns
        
        # Calculate ICT score
        ict_score = self._calculate_score(recent_mss, recent_fvg, df)
        
        # Determine bias
        bullish_signals = len([m for m in recent_mss if m.direction == "bullish"])
        bullish_signals += len([f for f in recent_fvg if f.direction == "bullish"])
        bearish_signals = len([m for m in recent_mss if m.direction == "bearish"])
        bearish_signals += len([f for f in recent_fvg if f.direction == "bearish"])
        
        if bullish_signals > bearish_signals:
            bias = "bullish"
        elif bearish_signals > bullish_signals:
            bias = "bearish"
        else:
            bias = "neutral"
        
        return {
            "mss_patterns": mss_patterns,
            "fvg_patterns": fvg_patterns,
            "recent_mss": recent_mss,
            "recent_fvg": recent_fvg,
            "has_mss": len(recent_mss) > 0,
            "has_fvg": len(recent_fvg) > 0,
            "ict_score": ict_score,
            "bias": bias
        }
    
    def _calculate_score(self, mss_list: List[MSS], fvg_list: List[FVG], df: pd.DataFrame) -> float:
        """Calculate ICT confluence score (0-100)"""
        score = 0
        
        # MSS contribution (max 40 points)
        if mss_list:
            best_mss = max(mss_list, key=lambda x: x.strength)
            score += min(best_mss.strength * 40, 40)
        
        # FVG contribution (max 30 points)
        if fvg_list:
            # Unfilled FVGs near current price are more valuable
            current_price = df["close"].iloc[-1]
            for fvg in fvg_list[-5:]:  # Last 5 FVGs
                distance_pct = abs((fvg.gap_high + fvg.gap_low) / 2 - current_price) / current_price * 100
                if distance_pct < 2:  # Within 2% of current price
                    score += 15
                    break
                elif distance_pct < 5:
                    score += 10
                    break
            
            # Multiple FVGs confluence
            if len(fvg_list) >= 2:
                score += 10
        
        # Direction confluence bonus (max 30 points)
        if mss_list and fvg_list:
            mss_direction = mss_list[-1].direction
            fvg_direction = fvg_list[-1].direction
            if mss_direction == fvg_direction:
                score += 30
        
        return min(score, 100)


def test_ict_detector():
    """Test ICT detector with sample data"""
    # Create sample OHLCV data
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=100, freq="4H")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices + np.random.randn(100) * 0.1,
        "high": prices + np.abs(np.random.randn(100) * 0.5),
        "low": prices - np.abs(np.random.randn(100) * 0.5),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 100)
    })
    
    detector = ICTDetector()
    result = detector.analyze(df)
    
    print(f"MSS Patterns Found: {len(result['mss_patterns'])}")
    print(f"FVG Patterns Found: {len(result['fvg_patterns'])}")
    print(f"ICT Score: {result['ict_score']:.1f}")
    print(f"Bias: {result['bias']}")


if __name__ == "__main__":
    test_ict_detector()

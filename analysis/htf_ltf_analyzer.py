"""
HTF/LTF Top-Down Analyzer
Multi-timeframe trend analysis using top-down approach
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class TrendDirection(Enum):
    """Trend Direction"""
    BULLISH = "상승 추세"
    BEARISH = "하락 추세"
    SIDEWAYS = "횡보"
    UNKNOWN = "미확정"


@dataclass
class TimeframeTrend:
    """Single timeframe trend analysis"""
    timeframe: str
    direction: TrendDirection
    strength: float  # 0-1
    key_level_support: float
    key_level_resistance: float
    structure: str  # "HH/HL" or "LH/LL" or "Range"
    description: str


class HTFLTFAnalyzer:
    """
    HTF/LTF Top-Down Analyzer
    
    Analyzes market structure across multiple timeframes:
    - HTF (Higher Timeframe): 1D, 1W - Overall trend direction
    - LTF (Lower Timeframe): 1H, 4H - Entry timing
    """
    
    def __init__(self, ema_periods: List[int] = [20, 50, 200]):
        """
        Initialize HTF/LTF Analyzer
        
        Args:
            ema_periods: EMA periods for trend detection
        """
        self.ema_periods = ema_periods
    
    def calculate_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs for trend analysis"""
        df = df.copy()
        for period in self.ema_periods:
            if len(df) >= period:
                df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        return df
    
    def detect_swing_structure(self, df: pd.DataFrame, lookback: int = 5) -> str:
        """
        Detect swing high/low structure
        
        Returns:
            "HH/HL" for bullish, "LH/LL" for bearish, "Range" for sideways
        """
        if len(df) < lookback * 4:
            return "Unknown"
        
        highs = df["high"].values
        lows = df["low"].values
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Swing high
            if highs[i] == max(highs[i-lookback:i+lookback+1]):
                swing_highs.append((i, highs[i]))
            # Swing low
            if lows[i] == min(lows[i-lookback:i+lookback+1]):
                swing_lows.append((i, lows[i]))
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "Range"
        
        # Check last two swing highs and lows
        last_two_highs = [sh[1] for sh in swing_highs[-2:]]
        last_two_lows = [sl[1] for sl in swing_lows[-2:]]
        
        hh = last_two_highs[-1] > last_two_highs[-2]
        hl = last_two_lows[-1] > last_two_lows[-2]
        lh = last_two_highs[-1] < last_two_highs[-2]
        ll = last_two_lows[-1] < last_two_lows[-2]
        
        if hh and hl:
            return "HH/HL"  # Bullish
        elif lh and ll:
            return "LH/LL"  # Bearish
        else:
            return "Range"
    
    def analyze_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> TimeframeTrend:
        """
        Analyze a single timeframe
        
        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe label (e.g., "1D", "4H")
            
        Returns:
            TimeframeTrend analysis
        """
        if len(df) < 50:
            return TimeframeTrend(
                timeframe=timeframe,
                direction=TrendDirection.UNKNOWN,
                strength=0,
                key_level_support=df["low"].min(),
                key_level_resistance=df["high"].max(),
                structure="Unknown",
                description=f"{timeframe} 데이터 부족"
            )
        
        df = self.calculate_emas(df)
        current_price = df["close"].iloc[-1]
        
        # Detect structure
        structure = self.detect_swing_structure(df)
        
        # EMA-based trend
        ema_20 = df[f"ema_20"].iloc[-1] if "ema_20" in df.columns else current_price
        ema_50 = df[f"ema_50"].iloc[-1] if "ema_50" in df.columns else current_price
        ema_200 = df[f"ema_200"].iloc[-1] if "ema_200" in df.columns else current_price
        
        # Trend strength (0-1)
        ema_alignment_bullish = (current_price > ema_20 > ema_50)
        ema_alignment_bearish = (current_price < ema_20 < ema_50)
        
        strength = 0.5
        if ema_alignment_bullish:
            strength = 0.7 + (0.3 if current_price > ema_200 else 0)
        elif ema_alignment_bearish:
            strength = 0.7 + (0.3 if current_price < ema_200 else 0)
        
        # Determine direction
        if structure == "HH/HL" and current_price > ema_50:
            direction = TrendDirection.BULLISH
            description = f"{timeframe} 상승 추세 (고점/저점 상승 + EMA 정배열)"
        elif structure == "LH/LL" and current_price < ema_50:
            direction = TrendDirection.BEARISH
            description = f"{timeframe} 하락 추세 (고점/저점 하락 + EMA 역배열)"
        elif current_price > ema_200:
            direction = TrendDirection.BULLISH
            description = f"{timeframe} 장기 상승 추세 (200 EMA 위)"
        elif current_price < ema_200:
            direction = TrendDirection.BEARISH
            description = f"{timeframe} 장기 하락 추세 (200 EMA 아래)"
        else:
            direction = TrendDirection.SIDEWAYS
            description = f"{timeframe} 횡보 구간"
        
        # Key levels
        recent = df.tail(50)
        support = recent["low"].min()
        resistance = recent["high"].max()
        
        return TimeframeTrend(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            key_level_support=support,
            key_level_resistance=resistance,
            structure=structure,
            description=description
        )
    
    def analyze(
        self,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
        df_1w: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Perform full HTF/LTF analysis
        
        Args:
            df_1h: 1-hour OHLCV
            df_4h: 4-hour OHLCV
            df_1d: 1-day OHLCV
            df_1w: 1-week OHLCV
            
        Returns:
            Dict with multi-timeframe analysis
        """
        results = {}
        
        # Analyze each timeframe
        if df_1w is not None and len(df_1w) > 20:
            results["1W"] = self.analyze_timeframe(df_1w, "1W")
        if df_1d is not None and len(df_1d) > 20:
            results["1D"] = self.analyze_timeframe(df_1d, "1D")
        if df_4h is not None and len(df_4h) > 20:
            results["4H"] = self.analyze_timeframe(df_4h, "4H")
        if df_1h is not None and len(df_1h) > 20:
            results["1H"] = self.analyze_timeframe(df_1h, "1H")
        
        # Determine overall bias
        htf_direction = None
        ltf_direction = None
        
        # HTF = 1D or 1W
        if "1D" in results:
            htf_direction = results["1D"].direction
        elif "1W" in results:
            htf_direction = results["1W"].direction
        
        # LTF = 4H or 1H
        if "4H" in results:
            ltf_direction = results["4H"].direction
        elif "1H" in results:
            ltf_direction = results["1H"].direction
        
        # Check alignment
        aligned = htf_direction == ltf_direction if htf_direction and ltf_direction else False
        
        # Generate summary
        if aligned:
            if htf_direction == TrendDirection.BULLISH:
                summary = "HTF/LTF 상승 정렬: 모든 타임프레임에서 상승 추세 확인"
                bias = "LONG"
                score = 80
            elif htf_direction == TrendDirection.BEARISH:
                summary = "HTF/LTF 하락 정렬: 모든 타임프레임에서 하락 추세 확인"
                bias = "SHORT"
                score = 20
            else:
                summary = "HTF/LTF 횡보 정렬: 명확한 추세 없음"
                bias = "NEUTRAL"
                score = 50
        else:
            if htf_direction == TrendDirection.BEARISH and ltf_direction == TrendDirection.BULLISH:
                summary = "역행 구조: HTF 하락 추세 내 LTF 반등 (단기 롱 가능, 저항 주의)"
                bias = "COUNTER_TREND_LONG"
                score = 55
            elif htf_direction == TrendDirection.BULLISH and ltf_direction == TrendDirection.BEARISH:
                summary = "역행 구조: HTF 상승 추세 내 LTF 조정 (매수 기회 대기)"
                bias = "PULLBACK_BUY"
                score = 65
            else:
                summary = "혼합 신호: 타임프레임 간 불일치"
                bias = "MIXED"
                score = 50
        
        return {
            "timeframes": results,
            "htf_direction": htf_direction,
            "ltf_direction": ltf_direction,
            "aligned": aligned,
            "bias": bias,
            "htf_ltf_score": score,
            "summary": summary
        }


def test_htf_ltf():
    """Test HTF/LTF Analyzer"""
    import numpy as np
    np.random.seed(42)
    
    # Generate sample data
    dates_4h = pd.date_range(start="2024-01-01", periods=200, freq="4H")
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    df_4h = pd.DataFrame({
        "timestamp": dates_4h,
        "open": prices + np.random.randn(200) * 0.2,
        "high": prices + np.abs(np.random.randn(200) * 1),
        "low": prices - np.abs(np.random.randn(200) * 1),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 200)
    })
    
    analyzer = HTFLTFAnalyzer()
    result = analyzer.analyze(df_4h=df_4h)
    
    print(f"HTF Direction: {result['htf_direction']}")
    print(f"LTF Direction: {result['ltf_direction']}")
    print(f"Aligned: {result['aligned']}")
    print(f"Bias: {result['bias']}")
    print(f"Score: {result['htf_ltf_score']}")
    print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    test_htf_ltf()

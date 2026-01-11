"""
VPA (Volume Price Analysis) Detector
Detects volume-based trend exhaustion, climax, and accumulation patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class VPASignal(Enum):
    """VPA Signal Types"""
    BUYING_CLIMAX = "Buying Climax"
    SELLING_CLIMAX = "Selling Climax"
    NO_DEMAND = "No Demand"
    NO_SUPPLY = "No Supply"
    STOPPING_VOLUME = "Stopping Volume"
    TEST = "Test"
    UPTHRUST = "Upthrust"
    SPRING = "Spring"
    NEUTRAL = "Neutral"


@dataclass
class VPACandle:
    """VPA Candle Analysis Result"""
    timestamp: pd.Timestamp
    signal: VPASignal
    spread: float  # High - Low
    volume_ratio: float  # vs 20-period average
    close_position: float  # 0-1 within candle range
    is_climax: bool
    description: str


class VPADetector:
    """
    Volume Price Analysis Detector
    
    Analyzes the relationship between price spread, close position, 
    and volume to identify smart money activity.
    
    Key Concepts:
    - Spread: Range of the candle (High - Low)
    - Volume: Effort put into the move
    - Close Position: Where price closed within the range
    - Result: The actual price movement
    """
    
    def __init__(
        self,
        volume_avg_period: int = 20,
        high_volume_threshold: float = 1.5,
        low_volume_threshold: float = 0.7,
        wide_spread_threshold: float = 1.3,
        narrow_spread_threshold: float = 0.7
    ):
        """
        Initialize VPA Detector
        
        Args:
            volume_avg_period: Period for average volume calculation
            high_volume_threshold: Multiplier for high volume detection
            low_volume_threshold: Multiplier for low volume detection
            wide_spread_threshold: Multiplier for wide spread detection
            narrow_spread_threshold: Multiplier for narrow spread detection
        """
        self.volume_avg_period = volume_avg_period
        self.high_volume_threshold = high_volume_threshold
        self.low_volume_threshold = low_volume_threshold
        self.wide_spread_threshold = wide_spread_threshold
        self.narrow_spread_threshold = narrow_spread_threshold
    
    def calculate_close_position(self, row: pd.Series) -> float:
        """
        Calculate where the close is within the candle range
        0 = closed at low, 1 = closed at high
        """
        spread = row["high"] - row["low"]
        if spread == 0:
            return 0.5
        return (row["close"] - row["low"]) / spread
    
    def analyze_candle(
        self,
        row: pd.Series,
        avg_volume: float,
        avg_spread: float,
        prev_close: float
    ) -> VPACandle:
        """
        Analyze a single candle using VPA principles
        """
        spread = row["high"] - row["low"]
        volume = row["volume"]
        close_pos = self.calculate_close_position(row)
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        spread_ratio = spread / avg_spread if avg_spread > 0 else 1
        
        is_up = row["close"] > prev_close
        is_high_volume = volume_ratio >= self.high_volume_threshold
        is_low_volume = volume_ratio <= self.low_volume_threshold
        is_wide_spread = spread_ratio >= self.wide_spread_threshold
        is_narrow_spread = spread_ratio <= self.narrow_spread_threshold
        
        # Determine VPA signal
        signal = VPASignal.NEUTRAL
        description = ""
        is_climax = False
        
        # Buying Climax: Wide spread up, high volume, but close in middle/low
        if is_up and is_wide_spread and is_high_volume and close_pos < 0.5:
            signal = VPASignal.BUYING_CLIMAX
            description = "상승 클라이맥스: 높은 거래량에 넓은 양봉이지만 상단에서 매도 압력"
            is_climax = True
        
        # Selling Climax: Wide spread down, high volume, but close in middle/high
        elif not is_up and is_wide_spread and is_high_volume and close_pos > 0.5:
            signal = VPASignal.SELLING_CLIMAX
            description = "하락 클라이맥스: 높은 거래량에 넓은 음봉이지만 하단에서 매수 압력"
            is_climax = True
        
        # No Demand: Up bar with narrow spread and low volume
        elif is_up and is_narrow_spread and is_low_volume:
            signal = VPASignal.NO_DEMAND
            description = "무수요: 상승하나 좁은 범위와 낮은 거래량으로 추가 상승 어려움"
        
        # No Supply: Down bar with narrow spread and low volume
        elif not is_up and is_narrow_spread and is_low_volume:
            signal = VPASignal.NO_SUPPLY
            description = "무공급: 하락하나 좁은 범위와 낮은 거래량으로 추가 하락 어려움"
        
        # Stopping Volume: High volume but narrow spread (absorption)
        elif is_high_volume and is_narrow_spread:
            signal = VPASignal.STOPPING_VOLUME
            description = "흡수 거래량: 높은 거래량이 좁은 범위에서 흡수됨 (전환 신호)"
            is_climax = True
        
        # Upthrust: Up then close near low with high volume
        elif is_up and close_pos < 0.3 and is_high_volume:
            signal = VPASignal.UPTHRUST
            description = "업스러스트: 상승 시도 후 되돌림으로 약세 신호"
        
        # Spring: Down then close near high
        elif not is_up and close_pos > 0.7 and is_high_volume:
            signal = VPASignal.SPRING
            description = "스프링: 하락 시도 후 강한 반등으로 강세 신호"
        
        return VPACandle(
            timestamp=row["timestamp"] if "timestamp" in row.index else row.name,
            signal=signal,
            spread=spread,
            volume_ratio=volume_ratio,
            close_position=close_pos,
            is_climax=is_climax,
            description=description
        )
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform full VPA analysis on OHLCV data
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dict with VPA analysis results
        """
        if len(df) < self.volume_avg_period + 1:
            return {
                "signals": [],
                "climax_count": 0,
                "bullish_signals": 0,
                "bearish_signals": 0,
                "vpa_score": 50,
                "trend_exhaustion": False,
                "summary": "데이터 부족"
            }
        
        # Calculate averages
        df = df.copy()
        df["spread"] = df["high"] - df["low"]
        df["avg_volume"] = df["volume"].rolling(self.volume_avg_period).mean()
        df["avg_spread"] = df["spread"].rolling(self.volume_avg_period).mean()
        
        signals = []
        for i in range(self.volume_avg_period, len(df)):
            row = df.iloc[i]
            prev_close = df.iloc[i-1]["close"]
            
            signal = self.analyze_candle(
                row,
                row["avg_volume"],
                row["avg_spread"],
                prev_close
            )
            signals.append(signal)
        
        # Analyze recent signals (last 20 candles)
        recent_signals = signals[-20:] if len(signals) >= 20 else signals
        
        climax_count = sum(1 for s in recent_signals if s.is_climax)
        bullish_signals = sum(1 for s in recent_signals if s.signal in [
            VPASignal.SELLING_CLIMAX, VPASignal.NO_SUPPLY, 
            VPASignal.STOPPING_VOLUME, VPASignal.SPRING
        ])
        bearish_signals = sum(1 for s in recent_signals if s.signal in [
            VPASignal.BUYING_CLIMAX, VPASignal.NO_DEMAND, VPASignal.UPTHRUST
        ])
        
        # Calculate VPA score (0-100)
        vpa_score = 50 + (bullish_signals - bearish_signals) * 5
        vpa_score = max(0, min(100, vpa_score))
        
        # Trend exhaustion detection
        trend_exhaustion = climax_count >= 2
        
        # Generate summary
        if bullish_signals > bearish_signals:
            summary = f"강세 우위: 매수 신호 {bullish_signals}개 > 매도 신호 {bearish_signals}개"
        elif bearish_signals > bullish_signals:
            summary = f"약세 우위: 매도 신호 {bearish_signals}개 > 매수 신호 {bullish_signals}개"
        else:
            summary = "중립: 명확한 방향성 없음"
        
        if trend_exhaustion:
            summary += " | ⚠️ 클라이맥스 감지로 추세 소진 가능성"
        
        return {
            "signals": signals,
            "recent_signals": recent_signals,
            "climax_count": climax_count,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "vpa_score": vpa_score,
            "trend_exhaustion": trend_exhaustion,
            "summary": summary,
            "last_signal": signals[-1] if signals else None
        }


def test_vpa_detector():
    """Test VPA detector"""
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
    
    detector = VPADetector()
    result = detector.analyze(df)
    
    print(f"VPA Score: {result['vpa_score']}")
    print(f"Climax Count: {result['climax_count']}")
    print(f"Bullish Signals: {result['bullish_signals']}")
    print(f"Bearish Signals: {result['bearish_signals']}")
    print(f"Trend Exhaustion: {result['trend_exhaustion']}")
    print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    test_vpa_detector()

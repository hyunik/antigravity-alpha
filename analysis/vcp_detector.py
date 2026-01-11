"""
VCP (Volatility Contraction Pattern) Detector
Detects Mark Minervini's VCP patterns for breakout setups
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class VCPPattern:
    """VCP Pattern detection result"""
    timestamp: pd.Timestamp
    contraction_count: int  # Number of price contractions (T1, T2, T3...)
    tightness_ratio: float  # Current range vs initial range
    volume_dryup_ratio: float  # Current volume vs average volume
    pivot_price: float  # Breakout pivot point
    base_low: float  # Pattern's lowest point
    strength: float  # 0-1 pattern quality score


class VCPDetector:
    """
    VCP (Volatility Contraction Pattern) Detector
    
    Based on Mark Minervini's pattern:
    - Price forms a series of contracting ranges (T1 > T2 > T3...)
    - Volume decreases toward the end of the base
    - Tight range near pivot point indicates readiness for breakout
    """
    
    def __init__(
        self,
        contraction_lookback: int = 5,
        min_contractions: int = 2,
        tightness_threshold: float = 0.5,
        volume_dryup_threshold: float = 0.7
    ):
        """
        Initialize VCP Detector
        
        Args:
            contraction_lookback: Candles to compare for each contraction
            min_contractions: Minimum number of contractions required
            tightness_threshold: Range must be less than this ratio of initial range
            volume_dryup_threshold: Volume must be less than this ratio of average
        """
        self.contraction_lookback = contraction_lookback
        self.min_contractions = min_contractions
        self.tightness_threshold = tightness_threshold
        self.volume_dryup_threshold = volume_dryup_threshold
    
    def calculate_range(self, df: pd.DataFrame, start: int, end: int) -> float:
        """Calculate price range for a period"""
        if start >= end or end > len(df):
            return 0
        segment = df.iloc[start:end]
        return segment["high"].max() - segment["low"].min()
    
    def detect_contractions(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Detect price range contractions
        
        Returns:
            List of (index, range) tuples for each contraction
        """
        if len(df) < self.contraction_lookback * 3:
            return []
        
        ranges = []
        step = self.contraction_lookback
        
        for i in range(0, len(df) - step, step):
            range_val = self.calculate_range(df, i, i + step)
            if range_val > 0:
                ranges.append((i + step, range_val))
        
        return ranges
    
    def detect_volume_dryup(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect volume dry-up pattern
        
        Returns:
            Tuple of (has_dryup, dryup_ratio)
        """
        if len(df) < 20:
            return False, 1.0
        
        avg_volume = df["volume"].iloc[:-5].mean()
        recent_volume = df["volume"].tail(5).mean()
        
        if avg_volume == 0:
            return False, 1.0
        
        dryup_ratio = recent_volume / avg_volume
        has_dryup = dryup_ratio <= self.volume_dryup_threshold
        
        return has_dryup, dryup_ratio
    
    def detect_tightness(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect price tightness (volatility contraction)
        
        Returns:
            Tuple of (is_tight, tightness_ratio)
        """
        if len(df) < 20:
            return False, 1.0
        
        # Calculate average range of previous candles
        recent = df.tail(20)
        avg_range = (recent["high"] - recent["low"]).mean()
        
        # Calculate current range (last 5 candles)
        current_high = df["high"].tail(5).max()
        current_low = df["low"].tail(5).min()
        current_range = current_high - current_low
        
        # Compare to average of last 20 candles total range
        initial_range = recent["high"].max() - recent["low"].min()
        
        if initial_range == 0:
            return False, 1.0
        
        tightness_ratio = current_range / initial_range
        is_tight = tightness_ratio <= self.tightness_threshold
        
        return is_tight, tightness_ratio
    
    def detect_vcp(self, df: pd.DataFrame) -> Optional[VCPPattern]:
        """
        Detect VCP pattern in OHLCV data
        
        Args:
            df: OHLCV DataFrame (at least 30 candles)
            
        Returns:
            VCPPattern if detected, None otherwise
        """
        if len(df) < 30:
            return None
        
        # Detect contractions
        ranges = self.detect_contractions(df)
        
        if len(ranges) < self.min_contractions:
            return None
        
        # Check if ranges are contracting
        contracting = True
        contraction_count = 0
        
        for i in range(1, len(ranges)):
            if ranges[i][1] < ranges[i-1][1]:
                contraction_count += 1
            else:
                contracting = False
                break
        
        if contraction_count < self.min_contractions - 1:
            return None
        
        # Check tightness
        is_tight, tightness_ratio = self.detect_tightness(df)
        
        # Check volume dry-up
        has_dryup, volume_ratio = self.detect_volume_dryup(df)
        
        # Calculate pattern metrics
        pivot_price = df["high"].tail(10).max()
        base_low = df["low"].tail(30).min()
        
        # Calculate strength score
        strength = self._calculate_strength(
            contraction_count, 
            tightness_ratio, 
            volume_ratio,
            is_tight,
            has_dryup
        )
        
        # Only return if we have a valid pattern
        if strength > 0.3:
            return VCPPattern(
                timestamp=df["timestamp"].iloc[-1] if "timestamp" in df.columns else df.index[-1],
                contraction_count=contraction_count + 1,
                tightness_ratio=tightness_ratio,
                volume_dryup_ratio=volume_ratio,
                pivot_price=pivot_price,
                base_low=base_low,
                strength=strength
            )
        
        return None
    
    def _calculate_strength(
        self,
        contraction_count: int,
        tightness_ratio: float,
        volume_ratio: float,
        is_tight: bool,
        has_dryup: bool
    ) -> float:
        """Calculate VCP pattern strength (0-1)"""
        strength = 0.0
        
        # Contraction count contribution (max 0.3)
        strength += min(contraction_count * 0.1, 0.3)
        
        # Tightness contribution (max 0.35)
        if is_tight:
            strength += 0.35 * (1 - tightness_ratio)
        else:
            strength += 0.15 * max(0, 1 - tightness_ratio)
        
        # Volume dry-up contribution (max 0.35)
        if has_dryup:
            strength += 0.35 * (1 - volume_ratio)
        else:
            strength += 0.15 * max(0, 1 - volume_ratio)
        
        return min(strength, 1.0)
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform full VCP analysis
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dict with analysis results
        """
        vcp = self.detect_vcp(df)
        is_tight, tightness_ratio = self.detect_tightness(df)
        has_dryup, volume_ratio = self.detect_volume_dryup(df)
        
        # Calculate VCP score
        vcp_score = 0
        if vcp:
            vcp_score = vcp.strength * 100
        elif is_tight or has_dryup:
            # Partial credit for some characteristics
            if is_tight:
                vcp_score += 20
            if has_dryup:
                vcp_score += 20
        
        return {
            "has_vcp": vcp is not None,
            "vcp_pattern": vcp,
            "is_tight": is_tight,
            "tightness_ratio": tightness_ratio,
            "has_volume_dryup": has_dryup,
            "volume_ratio": volume_ratio,
            "vcp_score": vcp_score,
            "pivot_price": vcp.pivot_price if vcp else df["high"].tail(10).max(),
            "base_low": vcp.base_low if vcp else df["low"].tail(30).min()
        }


def test_vcp_detector():
    """Test VCP detector with sample data"""
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=60, freq="4H")
    
    # Simulate VCP pattern with contracting ranges
    base = 100
    prices = np.concatenate([
        np.linspace(100, 110, 10),  # Initial rise
        np.linspace(110, 105, 5),   # First pullback (T1)
        np.linspace(105, 108, 5),   # Recovery
        np.linspace(108, 106, 5),   # Second pullback (T2) - smaller
        np.linspace(106, 107.5, 5), # Recovery
        np.linspace(107.5, 107, 5), # Third pullback (T3) - even smaller
        np.linspace(107, 108, 10),  # Tight consolidation
        np.linspace(108, 107.5, 10) + np.random.randn(10) * 0.2,  # Very tight
        np.linspace(107.5, 108, 5), # Setup for breakout
    ])[:60]
    
    # Decreasing volume toward the end
    volume = np.concatenate([
        np.random.randint(8000, 12000, 30),
        np.random.randint(4000, 7000, 15),
        np.random.randint(2000, 4000, 15),
    ])[:60]
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices + np.random.randn(60) * 0.1,
        "high": prices + np.abs(np.random.randn(60) * 0.5),
        "low": prices - np.abs(np.random.randn(60) * 0.5),
        "close": prices,
        "volume": volume
    })
    
    detector = VCPDetector()
    result = detector.analyze(df)
    
    print(f"Has VCP: {result['has_vcp']}")
    print(f"Is Tight: {result['is_tight']} (ratio: {result['tightness_ratio']:.2f})")
    print(f"Volume Dryup: {result['has_volume_dryup']} (ratio: {result['volume_ratio']:.2f})")
    print(f"VCP Score: {result['vcp_score']:.1f}")
    print(f"Pivot Price: ${result['pivot_price']:.2f}")
    
    if result['vcp_pattern']:
        vcp = result['vcp_pattern']
        print(f"Contractions: {vcp.contraction_count}")
        print(f"Pattern Strength: {vcp.strength:.2f}")


if __name__ == "__main__":
    test_vcp_detector()

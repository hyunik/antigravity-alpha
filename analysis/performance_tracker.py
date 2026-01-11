"""
Performance Tracker
Tracks and evaluates the performance of trading recommendations
"""

import asyncio
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger


@dataclass
class RecommendationRecord:
    """Single recommendation record for tracking"""
    symbol: str
    timestamp: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_zone: Tuple[float, float]
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    total_score: float
    conviction: str
    
    # Performance tracking (filled later)
    actual_entry_price: Optional[float] = None
    current_price: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    hit_tp1: bool = False
    hit_tp2: bool = False
    hit_tp3: bool = False
    hit_sl: bool = False
    pnl_percent: float = 0.0
    status: str = "OPEN"  # "OPEN", "TP1", "TP2", "TP3", "SL", "EXPIRED"


@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    total_recommendations: int = 0
    total_wins: int = 0
    total_losses: int = 0
    win_rate: float = 0.0
    
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    
    hit_tp1_count: int = 0
    hit_tp2_count: int = 0
    hit_tp3_count: int = 0
    hit_sl_count: int = 0
    
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    
    best_trade: str = ""
    best_trade_pnl: float = 0.0
    worst_trade: str = ""
    worst_trade_pnl: float = 0.0
    
    by_conviction: Dict[str, Dict] = field(default_factory=dict)
    by_direction: Dict[str, Dict] = field(default_factory=dict)


class PerformanceTracker:
    """
    Performance Tracker for trading recommendations
    
    Features:
    - Store recommendations with entry/exit levels
    - Track actual price performance over time
    - Calculate win rate, P&L, and hit rates
    - Generate performance reports
    """
    
    DATA_DIR = Path("data/performance")
    
    def __init__(self):
        self.recommendations: List[RecommendationRecord] = []
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def add_recommendation(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_zone: Tuple[float, float],
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: float,
        total_score: float,
        conviction: str,
        actual_entry: Optional[float] = None
    ) -> RecommendationRecord:
        """Add a new recommendation to track"""
        rec = RecommendationRecord(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            direction=direction,
            entry_price=entry_price,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            total_score=total_score,
            conviction=conviction,
            actual_entry_price=actual_entry or entry_price
        )
        self.recommendations.append(rec)
        self.save_recommendations()
        return rec
    
    def update_prices(self, price_map: Dict[str, Dict]) -> None:
        """
        Update current prices and check for TP/SL hits
        
        Args:
            price_map: Dict of {symbol: {"current": price, "high": high, "low": low}}
        """
        for rec in self.recommendations:
            if rec.status not in ["OPEN"]:
                continue
            
            if rec.symbol not in price_map:
                continue
            
            data = price_map[rec.symbol]
            rec.current_price = data.get("current", rec.current_price)
            
            # Track max/min since entry
            high = data.get("high", rec.current_price)
            low = data.get("low", rec.current_price)
            
            if rec.max_price is None or high > rec.max_price:
                rec.max_price = high
            if rec.min_price is None or low < rec.min_price:
                rec.min_price = low
            
            # Check TP/SL hits based on direction
            if rec.direction == "LONG":
                self._check_long_targets(rec, high, low)
            else:
                self._check_short_targets(rec, high, low)
        
        self.save_recommendations()
    
    def _check_long_targets(self, rec: RecommendationRecord, high: float, low: float) -> None:
        """Check TP/SL for long positions"""
        entry = rec.actual_entry_price or rec.entry_price
        
        # Check SL first (worst case)
        if low <= rec.stop_loss:
            rec.hit_sl = True
            rec.status = "SL"
            rec.pnl_percent = ((rec.stop_loss - entry) / entry) * 100
            return
        
        # Check TPs (best case first)
        if high >= rec.take_profit_3 and not rec.hit_tp3:
            rec.hit_tp3 = True
            rec.hit_tp2 = True
            rec.hit_tp1 = True
            rec.status = "TP3"
            rec.pnl_percent = ((rec.take_profit_3 - entry) / entry) * 100
        elif high >= rec.take_profit_2 and not rec.hit_tp2:
            rec.hit_tp2 = True
            rec.hit_tp1 = True
            rec.status = "TP2"
            rec.pnl_percent = ((rec.take_profit_2 - entry) / entry) * 100
        elif high >= rec.take_profit_1 and not rec.hit_tp1:
            rec.hit_tp1 = True
            rec.status = "TP1"
            rec.pnl_percent = ((rec.take_profit_1 - entry) / entry) * 100
        else:
            # Still open - calculate unrealized P&L
            rec.pnl_percent = ((rec.current_price - entry) / entry) * 100 if rec.current_price else 0
    
    def _check_short_targets(self, rec: RecommendationRecord, high: float, low: float) -> None:
        """Check TP/SL for short positions"""
        entry = rec.actual_entry_price or rec.entry_price
        
        # Check SL first (worst case)
        if high >= rec.stop_loss:
            rec.hit_sl = True
            rec.status = "SL"
            rec.pnl_percent = ((entry - rec.stop_loss) / entry) * 100
            return
        
        # Check TPs for shorts (price going down)
        if low <= rec.take_profit_3 and not rec.hit_tp3:
            rec.hit_tp3 = True
            rec.hit_tp2 = True
            rec.hit_tp1 = True
            rec.status = "TP3"
            rec.pnl_percent = ((entry - rec.take_profit_3) / entry) * 100
        elif low <= rec.take_profit_2 and not rec.hit_tp2:
            rec.hit_tp2 = True
            rec.hit_tp1 = True
            rec.status = "TP2"
            rec.pnl_percent = ((entry - rec.take_profit_2) / entry) * 100
        elif low <= rec.take_profit_1 and not rec.hit_tp1:
            rec.hit_tp1 = True
            rec.status = "TP1"
            rec.pnl_percent = ((entry - rec.take_profit_1) / entry) * 100
        else:
            rec.pnl_percent = ((entry - rec.current_price) / entry) * 100 if rec.current_price else 0
    
    def expire_old_recommendations(self, days: int = 30) -> int:
        """Expire recommendations older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        expired_count = 0
        
        for rec in self.recommendations:
            if rec.status == "OPEN":
                rec_date = datetime.fromisoformat(rec.timestamp.replace('Z', '+00:00').replace('+00:00', ''))
                if rec_date < cutoff:
                    rec.status = "EXPIRED"
                    expired_count += 1
        
        if expired_count > 0:
            self.save_recommendations()
        
        return expired_count
    
    def calculate_stats(self, days: Optional[int] = None) -> PerformanceStats:
        """Calculate performance statistics"""
        stats = PerformanceStats()
        
        # Filter by date if specified
        recs = self.recommendations
        if days:
            cutoff = datetime.utcnow() - timedelta(days=days)
            recs = [
                r for r in recs 
                if datetime.fromisoformat(r.timestamp.replace('Z', '+00:00').replace('+00:00', '')) >= cutoff
            ]
        
        if not recs:
            return stats
        
        stats.total_recommendations = len(recs)
        
        wins = []
        losses = []
        
        for rec in recs:
            if rec.hit_tp1 or rec.hit_tp2 or rec.hit_tp3:
                stats.total_wins += 1
                wins.append(rec.pnl_percent)
            elif rec.hit_sl:
                stats.total_losses += 1
                losses.append(rec.pnl_percent)
            
            if rec.hit_tp1:
                stats.hit_tp1_count += 1
            if rec.hit_tp2:
                stats.hit_tp2_count += 1
            if rec.hit_tp3:
                stats.hit_tp3_count += 1
            if rec.hit_sl:
                stats.hit_sl_count += 1
            
            stats.total_pnl_pct += rec.pnl_percent
        
        # Calculate rates
        closed = stats.total_wins + stats.total_losses
        stats.win_rate = (stats.total_wins / closed * 100) if closed > 0 else 0
        stats.avg_pnl_pct = stats.total_pnl_pct / len(recs) if recs else 0
        
        if wins:
            stats.avg_win_pct = sum(wins) / len(wins)
        if losses:
            stats.avg_loss_pct = sum(losses) / len(losses)
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Best/Worst trades
        all_pnls = [(r.symbol, r.pnl_percent) for r in recs if r.status != "OPEN"]
        if all_pnls:
            best = max(all_pnls, key=lambda x: x[1])
            worst = min(all_pnls, key=lambda x: x[1])
            stats.best_trade = best[0]
            stats.best_trade_pnl = best[1]
            stats.worst_trade = worst[0]
            stats.worst_trade_pnl = worst[1]
        
        # By conviction
        for conv in ["HIGH", "MEDIUM", "LOW"]:
            conv_recs = [r for r in recs if r.conviction == conv]
            if conv_recs:
                conv_wins = len([r for r in conv_recs if r.hit_tp1 or r.hit_tp2 or r.hit_tp3])
                conv_total = len([r for r in conv_recs if r.status != "OPEN"])
                stats.by_conviction[conv] = {
                    "count": len(conv_recs),
                    "wins": conv_wins,
                    "win_rate": (conv_wins / conv_total * 100) if conv_total > 0 else 0,
                    "avg_pnl": sum(r.pnl_percent for r in conv_recs) / len(conv_recs)
                }
        
        # By direction
        for direction in ["LONG", "SHORT"]:
            dir_recs = [r for r in recs if r.direction == direction]
            if dir_recs:
                dir_wins = len([r for r in dir_recs if r.hit_tp1 or r.hit_tp2 or r.hit_tp3])
                dir_total = len([r for r in dir_recs if r.status != "OPEN"])
                stats.by_direction[direction] = {
                    "count": len(dir_recs),
                    "wins": dir_wins,
                    "win_rate": (dir_wins / dir_total * 100) if dir_total > 0 else 0,
                    "avg_pnl": sum(r.pnl_percent for r in dir_recs) / len(dir_recs)
                }
        
        return stats
    
    def save_recommendations(self) -> None:
        """Save recommendations to file"""
        filepath = self.DATA_DIR / "recommendations.json"
        data = [asdict(r) for r in self.recommendations]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_recommendations(self) -> None:
        """Load recommendations from file"""
        filepath = self.DATA_DIR / "recommendations.json"
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
                self.recommendations = []
                for item in data:
                    # Convert entry_zone back to tuple
                    item["entry_zone"] = tuple(item["entry_zone"])
                    self.recommendations.append(RecommendationRecord(**item))
    
    def get_top_performers(self, n: int = 15) -> List[RecommendationRecord]:
        """Get top N performing recommendations"""
        sorted_recs = sorted(
            [r for r in self.recommendations if r.status != "OPEN"],
            key=lambda x: x.pnl_percent,
            reverse=True
        )
        return sorted_recs[:n]
    
    def get_recent_recommendations(self, n: int = 15) -> List[RecommendationRecord]:
        """Get most recent N recommendations"""
        sorted_recs = sorted(
            self.recommendations,
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_recs[:n]


def format_performance_report(stats: PerformanceStats, recs: List[RecommendationRecord], period_days: int = 30) -> str:
    """Format performance report as text"""
    
    # Format individual recommendations table
    rec_table = ""
    for i, rec in enumerate(recs[:15], 1):
        status_emoji = {
            "TP3": "🏆", "TP2": "✅", "TP1": "✅",
            "SL": "❌", "OPEN": "⏳", "EXPIRED": "⌛"
        }.get(rec.status, "")
        
        pnl_str = f"{rec.pnl_percent:+.2f}%" if rec.pnl_percent != 0 else "0.00%"
        
        rec_table += f"│ {i:2d} │ {rec.symbol:10s} │ {rec.direction:5s} │ {rec.total_score:5.0f} │ {rec.conviction:6s} │ {status_emoji} {rec.status:7s} │ {pnl_str:>8s} │\n"
    
    # Conviction breakdown
    conv_text = ""
    for conv in ["HIGH", "MEDIUM", "LOW"]:
        if conv in stats.by_conviction:
            d = stats.by_conviction[conv]
            conv_text += f"  {conv:6s}: {d['count']:3d}건 | 승률 {d['win_rate']:5.1f}% | 평균 P&L {d['avg_pnl']:+.2f}%\n"
    
    # Direction breakdown
    dir_text = ""
    for direction in ["LONG", "SHORT"]:
        if direction in stats.by_direction:
            d = stats.by_direction[direction]
            dir_text += f"  {direction:5s}: {d['count']:3d}건 | 승률 {d['win_rate']:5.1f}% | 평균 P&L {d['avg_pnl']:+.2f}%\n"
    
    report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 성과 분석 보고서 (최근 {period_days}일)
📅 생성 시간: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


━━━━━━━━━━━━━━━━ [1] 전체 성과 요약 ━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────┐
│                     핵심 성과 지표                          │
├─────────────────────────────────────────────────────────────┤
│  총 추천 건수:     {stats.total_recommendations:5d}건                                │
│  승리 (TP 도달):   {stats.total_wins:5d}건                                │
│  패배 (SL 도달):   {stats.total_losses:5d}건                                │
│  ───────────────────────────────────────────────────        │
│  승률(Win Rate):   {stats.win_rate:5.1f}%                                 │
│  이익 비율(PF):    {stats.profit_factor:5.2f}x                                 │
├─────────────────────────────────────────────────────────────┤
│  총 누적 P&L:      {stats.total_pnl_pct:+6.2f}%                               │
│  평균 P&L:         {stats.avg_pnl_pct:+6.2f}%                               │
│  평균 승리 P&L:    {stats.avg_win_pct:+6.2f}%                               │
│  평균 패배 P&L:    {stats.avg_loss_pct:+6.2f}%                               │
└─────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━ [2] 목표가 도달률 ━━━━━━━━━━━━━━━━

  ✅ TP1 도달: {stats.hit_tp1_count:3d}건 ({stats.hit_tp1_count/stats.total_recommendations*100 if stats.total_recommendations > 0 else 0:.1f}%)
  ✅ TP2 도달: {stats.hit_tp2_count:3d}건 ({stats.hit_tp2_count/stats.total_recommendations*100 if stats.total_recommendations > 0 else 0:.1f}%)
  🏆 TP3 도달: {stats.hit_tp3_count:3d}건 ({stats.hit_tp3_count/stats.total_recommendations*100 if stats.total_recommendations > 0 else 0:.1f}%)
  ❌ SL 도달:  {stats.hit_sl_count:3d}건 ({stats.hit_sl_count/stats.total_recommendations*100 if stats.total_recommendations > 0 else 0:.1f}%)


━━━━━━━━━━━━━━━━ [3] 신뢰도별 성과 ━━━━━━━━━━━━━━━━

{conv_text if conv_text else '  데이터 없음'}

━━━━━━━━━━━━━━━━ [4] 방향별 성과 ━━━━━━━━━━━━━━━━

{dir_text if dir_text else '  데이터 없음'}

━━━━━━━━━━━━━━━━ [5] 베스트/워스트 트레이드 ━━━━━━━━━━━━━━━━

  🏆 최고 성과: {stats.best_trade} ({stats.best_trade_pnl:+.2f}%)
  💔 최저 성과: {stats.worst_trade} ({stats.worst_trade_pnl:+.2f}%)


━━━━━━━━━━━━━━━━ [6] Top 15 추천 상세 ━━━━━━━━━━━━━━━━

┌────┬────────────┬───────┬───────┬────────┬───────────┬──────────┐
│ #  │ 코인       │ 방향  │ 점수  │ 신뢰도 │ 상태      │ P&L      │
├────┼────────────┼───────┼───────┼────────┼───────────┼──────────┤
{rec_table}└────┴────────────┴───────┴───────┴────────┴───────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 분석 인사이트:
{_generate_insights(stats)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report


def _generate_insights(stats: PerformanceStats) -> str:
    """Generate performance insights"""
    insights = []
    
    if stats.win_rate >= 60:
        insights.append(f"✅ 승률 {stats.win_rate:.1f}%로 양호한 성과를 보이고 있습니다.")
    elif stats.win_rate >= 40:
        insights.append(f"⚠️ 승률이 {stats.win_rate:.1f}%로 개선이 필요합니다.")
    else:
        insights.append(f"❌ 승률이 {stats.win_rate:.1f}%로 전략 재검토가 필요합니다.")
    
    if stats.profit_factor >= 2.0:
        insights.append(f"✅ Profit Factor {stats.profit_factor:.2f}x로 수익 구조가 매우 우수합니다.")
    elif stats.profit_factor >= 1.0:
        insights.append(f"⚠️ Profit Factor {stats.profit_factor:.2f}x로 수익은 나고 있으나 개선 여지가 있습니다.")
    else:
        insights.append(f"❌ Profit Factor {stats.profit_factor:.2f}x로 전략 수정이 필요합니다.")
    
    # Conviction analysis
    if "HIGH" in stats.by_conviction and "LOW" in stats.by_conviction:
        high_wr = stats.by_conviction["HIGH"].get("win_rate", 0)
        low_wr = stats.by_conviction["LOW"].get("win_rate", 0)
        if high_wr > low_wr + 10:
            insights.append("✅ HIGH 신뢰도 추천이 LOW보다 성과가 좋습니다. 신뢰도 시스템이 유효합니다.")
        elif low_wr > high_wr:
            insights.append("⚠️ LOW 신뢰도가 HIGH보다 성과가 좋습니다. 신뢰도 기준 재검토 필요합니다.")
    
    return "\n".join(f"  • {i}" for i in insights) if insights else "  • 충분한 데이터가 축적되면 인사이트가 생성됩니다."


async def test_performance_tracker():
    """Test performance tracker with sample data"""
    tracker = PerformanceTracker()
    
    # Add sample recommendations
    sample_data = [
        ("BTCUSDT", "LONG", 90000, (89500, 90500), 87000, 92000, 95000, 100000, 75, "HIGH"),
        ("ETHUSDT", "LONG", 3000, (2950, 3050), 2800, 3200, 3500, 3800, 68, "MEDIUM"),
        ("WIFUSDT", "LONG", 0.28, (0.27, 0.29), 0.25, 0.32, 0.35, 0.40, 62, "MEDIUM"),
        ("SOLUSDT", "SHORT", 200, (195, 205), 220, 180, 160, 140, 55, "LOW"),
    ]
    
    for data in sample_data:
        tracker.add_recommendation(*data)
    
    # Simulate price updates
    tracker.update_prices({
        "BTCUSDT": {"current": 93000, "high": 94000, "low": 89000},
        "ETHUSDT": {"current": 3100, "high": 3250, "low": 2950},
        "WIFUSDT": {"current": 0.30, "high": 0.33, "low": 0.26},
        "SOLUSDT": {"current": 185, "high": 210, "low": 175},
    })
    
    # Calculate stats
    stats = tracker.calculate_stats(days=30)
    recs = tracker.get_recent_recommendations(15)
    
    # Print report
    print(format_performance_report(stats, recs, 30))


if __name__ == "__main__":
    asyncio.run(test_performance_tracker())

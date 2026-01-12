"""
Discord Report Generator
Creates formatted reports for Discord distribution
"""

from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

from agents.cio_agent import TradeRecommendation


@dataclass
class DiscordEmbed:
    """Discord embed structure"""
    title: str
    description: str
    color: int
    fields: List[Dict]
    footer: str
    timestamp: str


class ReportGenerator:
    """
    Generates formatted reports for Discord
    
    Creates rich embeds with:
    - Trade setup summary
    - Entry/exit levels
    - Key reasons and risks
    - Visual formatting
    """
    
    # Color codes for different conviction levels
    COLORS = {
        "HIGH": 0x00FF00,      # Green
        "MEDIUM": 0xFFFF00,    # Yellow
        "LOW": 0xFF6600,       # Orange
    }
    
    # Direction emojis
    DIRECTION_EMOJI = {
        "LONG": "🟢 LONG",
        "SHORT": "🔴 SHORT",
        "NEUTRAL": "⚪ NEUTRAL"
    }
    
    def format_price(self, price: float) -> str:
        """Format price with appropriate decimal places"""
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 1:
            return f"${price:.4f}"
        else:
            return f"${price:.6f}"
    
    def format_percent(self, percent: float) -> str:
        """Format percentage"""
        return f"{percent:.2f}%"
    
    def generate_single_report(self, rec: TradeRecommendation) -> DiscordEmbed:
        """
        Generate Discord embed for a single trade recommendation
        
        Args:
            rec: TradeRecommendation from CIO Agent
            
        Returns:
            DiscordEmbed ready for sending
        """
        direction_text = self.DIRECTION_EMOJI.get(rec.direction, rec.direction)
        conviction_emoji = "🔥" if rec.conviction == "HIGH" else "⚡" if rec.conviction == "MEDIUM" else "💡"
        
        # Title with symbol and direction
        title = f"{conviction_emoji} {rec.symbol} | {direction_text}"
        
        # Description with current price and score
        description = (
            f"**현재가:** {self.format_price(rec.current_price)}\n"
            f"**신뢰도:** {rec.conviction} ({rec.score:.1f}/100)\n"
            f"**분석 시간:** {rec.timestamp[:16].replace('T', ' ')} UTC"
        )
        
        # Build fields
        fields = []
        
        # Entry Zone
        fields.append({
            "name": "📍 진입 구간",
            "value": f"{self.format_price(rec.entry_zone[0])} - {self.format_price(rec.entry_zone[1])}",
            "inline": True
        })
        
        # Stop Loss
        fields.append({
            "name": "🛑 손절가",
            "value": f"{self.format_price(rec.stop_loss)} ({self.format_percent(rec.risk_percent)} risk)",
            "inline": True
        })
        
        # Empty field for spacing
        fields.append({
            "name": "\u200b",
            "value": "\u200b",
            "inline": True
        })
        
        # Targets
        targets_text = (
            f"**TP1 (1:1):** {self.format_price(rec.target_1)} (+{self.format_percent(rec.reward_1_percent)})\n"
            f"**TP2 (1:2):** {self.format_price(rec.target_2)} (+{self.format_percent(rec.reward_2_percent)})\n"
            f"**TP3 (1:3):** {self.format_price(rec.target_3)} (+{self.format_percent(rec.reward_3_percent)})"
        )
        fields.append({
            "name": "🎯 목표가",
            "value": targets_text,
            "inline": False
        })
        
        # Key Reasons
        if rec.key_reasons:
            reasons_text = "\n".join(f"• {r}" for r in rec.key_reasons[:5])
            fields.append({
                "name": "📊 핵심 근거",
                "value": reasons_text,
                "inline": False
            })
        
        # Risk Factors
        if rec.risk_factors:
            risks_text = "\n".join(f"⚠️ {r}" for r in rec.risk_factors[:3])
            fields.append({
                "name": "⚠️ 리스크 요인",
                "value": risks_text,
                "inline": False
            })
        
        # Agent Reasoning (truncated)
        if rec.agent_reasoning:
            reasoning = rec.agent_reasoning[:800]
            if len(rec.agent_reasoning) > 800:
                reasoning += "..."
            fields.append({
                "name": "🤖 AI 분석",
                "value": reasoning,
                "inline": False
            })
        
        return DiscordEmbed(
            title=title,
            description=description,
            color=self.COLORS.get(rec.conviction, 0x808080),
            fields=fields,
            footer="Antigravity-Alpha | Smart Coin Select",
            timestamp=rec.timestamp
        )
    
    def generate_summary_report(
        self,
        recommendations: List[TradeRecommendation],
        total_coins_analyzed: int
    ) -> DiscordEmbed:
        """
        Generate summary report for multiple recommendations
        
        Args:
            recommendations: List of TradeRecommendation
            total_coins_analyzed: Total number of coins analyzed
            
        Returns:
            DiscordEmbed with summary
        """
        now = datetime.utcnow().isoformat()
        
        # Count by direction
        longs = len([r for r in recommendations if r.direction == "LONG"])
        shorts = len([r for r in recommendations if r.direction == "SHORT"])
        
        # Count by conviction
        high_conviction = len([r for r in recommendations if r.conviction == "HIGH"])
        medium_conviction = len([r for r in recommendations if r.conviction == "MEDIUM"])
        
        title = "📈 Antigravity-Alpha 일일 리포트"
        
        description = (
            f"**분석 완료:** {total_coins_analyzed}개 코인 분석\n"
            f"**시그널 발생:** {len(recommendations)}개 매매 기회 감지\n"
            f"**분석 시간:** {now[:16].replace('T', ' ')} UTC"
        )
        
        fields = []
        
        # Direction breakdown
        fields.append({
            "name": "📊 방향별 분포",
            "value": f"🟢 LONG: {longs}개 | 🔴 SHORT: {shorts}개",
            "inline": False
        })
        
        # Conviction breakdown
        fields.append({
            "name": "🎯 신뢰도별 분포",
            "value": f"🔥 HIGH: {high_conviction}개 | ⚡ MEDIUM: {medium_conviction}개",
            "inline": False
        })
        
        # Top picks (high conviction)
        high_picks = [r for r in recommendations if r.conviction == "HIGH"][:5]
        if high_picks:
            picks_text = "\n".join(
                f"{i+1}. **{r.symbol}** | {self.DIRECTION_EMOJI[r.direction]} | Score: {r.score:.1f}"
                for i, r in enumerate(high_picks)
            )
            fields.append({
                "name": "🔥 Top Picks (High Conviction)",
                "value": picks_text,
                "inline": False
            })
        
        return DiscordEmbed(
            title=title,
            description=description,
            color=0x5865F2,  # Discord blurple
            fields=fields,
            footer="Antigravity-Alpha | Powered by ICT/Wyckoff/VCP Analysis",
            timestamp=now
        )
    
    def to_discord_payload(self, embed: DiscordEmbed) -> Dict:
        """
        Convert DiscordEmbed to Discord API payload format
        
        Args:
            embed: DiscordEmbed object
            
        Returns:
            Dict ready for Discord webhook
        """
        embed_dict = {
            "title": embed.title,
            "description": embed.description,
            "color": embed.color,
            "fields": embed.fields,
            "footer": {"text": embed.footer},
            "timestamp": embed.timestamp
        }
        
        return {"embeds": [embed_dict]}
    
    def generate_batch_payload(
        self,
        recommendations: List[TradeRecommendation],
        total_analyzed: int,
        max_embeds: int = 10
    ) -> List[Dict]:
        """
        Generate batch of Discord payloads
        
        Args:
            recommendations: List of recommendations
            total_analyzed: Total coins analyzed
            max_embeds: Maximum embeds per message (Discord limit: 10)
            
        Returns:
            List of Discord API payloads
        """
        payloads = []
        
        # Summary as first message
        summary = self.generate_summary_report(recommendations, total_analyzed)
        payloads.append(self.to_discord_payload(summary))
        
        # Individual recommendations
        for rec in recommendations[:max_embeds]:
            embed = self.generate_single_report(rec)
            payloads.append(self.to_discord_payload(embed))
        
        return payloads


def test_report_generator():
    """Test report generator"""
    from agents.cio_agent import TradeRecommendation
    
    mock_rec = TradeRecommendation(
        symbol="BTCUSDT",
        direction="LONG",
        conviction="HIGH",
        score=85.5,
        current_price=43500.0,
        entry_zone=(42500.0, 43200.0),
        stop_loss=41500.0,
        target_1=45000.0,
        target_2=48000.0,
        target_3=52000.0,
        risk_percent=4.5,
        reward_1_percent=4.2,
        reward_2_percent=11.5,
        reward_3_percent=20.3,
        key_reasons=["Bullish MSS detected", "4h FVG", "Wyckoff Accumulation Phase", "VCP tightness"],
        risk_factors=["High funding rate - potential squeeze"],
        agent_reasoning="Analysis shows strong bullish confluence...",
        timestamp=datetime.utcnow().isoformat()
    )
    
    generator = ReportGenerator()
    embed = generator.generate_single_report(mock_rec)
    payload = generator.to_discord_payload(embed)
    
    print("Discord Embed Generated:")
    print(f"Title: {embed.title}")
    print(f"Color: {hex(embed.color)}")
    print(f"Fields: {len(embed.fields)}")
    
    import json
    print("\nPayload Preview:")
    print(json.dumps(payload, indent=2, default=str)[:1000])


if __name__ == "__main__":
    test_report_generator()

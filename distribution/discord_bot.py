"""
Discord Bot
Sends formatted reports to Discord via webhook
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from config.settings import settings
from agents.cio_agent import TradeRecommendation
from .report_generator import ReportGenerator


class DiscordBot:
    """
    Discord bot for sending trading reports via webhook
    
    Features:
    - Send individual trade recommendations
    - Send summary reports
    - Rate limiting to avoid Discord limits
    - Retry logic for failed sends
    """
    
    RATE_LIMIT_DELAY = 1.0  # Seconds between messages
    MAX_RETRIES = 3
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord bot
        
        Args:
            webhook_url: Discord webhook URL (optional, uses settings if not provided)
        """
        self.webhook_url = webhook_url or settings.discord.webhook_url
        self.report_generator = ReportGenerator()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _send_webhook(self, payload: Dict, retries: int = 0) -> bool:
        """
        Send payload to Discord webhook
        
        Args:
            payload: Discord API payload
            retries: Current retry count
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            logger.error("Discord webhook URL not configured")
            return False
        
        session = await self._get_session()
        
        try:
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status == 204:
                    return True
                elif response.status == 429:
                    # Rate limited
                    retry_after = response.headers.get("Retry-After", "5")
                    logger.warning(f"Discord rate limited, waiting {retry_after}s")
                    await asyncio.sleep(float(retry_after))
                    if retries < self.MAX_RETRIES:
                        return await self._send_webhook(payload, retries + 1)
                else:
                    text = await response.text()
                    logger.error(f"Discord webhook failed: {response.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
            if retries < self.MAX_RETRIES:
                await asyncio.sleep(2 ** retries)
                return await self._send_webhook(payload, retries + 1)
            return False
        
        return False
    
    async def send_recommendation(self, rec: TradeRecommendation) -> bool:
        """
        Send a single trade recommendation to Discord
        
        Args:
            rec: TradeRecommendation to send
            
        Returns:
            True if successful
        """
        embed = self.report_generator.generate_single_report(rec)
        payload = self.report_generator.to_discord_payload(embed)
        
        success = await self._send_webhook(payload)
        if success:
            logger.info(f"Sent recommendation for {rec.symbol} to Discord")
        else:
            logger.error(f"Failed to send recommendation for {rec.symbol}")
        
        return success
    
    async def send_summary(
        self,
        recommendations: List[TradeRecommendation],
        total_analyzed: int
    ) -> bool:
        """
        Send summary report to Discord
        
        Args:
            recommendations: List of recommendations
            total_analyzed: Total coins analyzed
            
        Returns:
            True if successful
        """
        embed = self.report_generator.generate_summary_report(recommendations, total_analyzed)
        payload = self.report_generator.to_discord_payload(embed)
        
        success = await self._send_webhook(payload)
        if success:
            logger.info("Sent summary report to Discord")
        else:
            logger.error("Failed to send summary report")
        
        return success
    
    async def send_batch(
        self,
        recommendations: List[TradeRecommendation],
        total_analyzed: int,
        include_summary: bool = True,
        max_recommendations: int = 10
    ) -> Dict[str, int]:
        """
        Send batch of recommendations to Discord
        
        Args:
            recommendations: List of recommendations
            total_analyzed: Total coins analyzed
            include_summary: Whether to include summary report
            max_recommendations: Maximum individual recommendations to send
            
        Returns:
            Dict with success/failure counts
        """
        results = {"success": 0, "failed": 0}
        
        # Send summary first
        if include_summary:
            if await self.send_summary(recommendations, total_analyzed):
                results["success"] += 1
            else:
                results["failed"] += 1
            await asyncio.sleep(self.RATE_LIMIT_DELAY)
        
        # Send individual recommendations
        for rec in recommendations[:max_recommendations]:
            if await self.send_recommendation(rec):
                results["success"] += 1
            else:
                results["failed"] += 1
            
            await asyncio.sleep(self.RATE_LIMIT_DELAY)
        
        logger.info(f"Batch send complete: {results['success']} success, {results['failed']} failed")
        return results
    
    async def send_alert(self, title: str, message: str, color: int = 0xFF0000) -> bool:
        """
        Send a simple alert message
        
        Args:
            title: Alert title
            message: Alert message
            color: Embed color (default: red)
            
        Returns:
            True if successful
        """
        payload = {
            "embeds": [{
                "title": f"⚠️ {title}",
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        return await self._send_webhook(payload)
    
    async def send_text(self, content: str) -> bool:
        """
        Send a simple text message
        
        Args:
            content: Text content to send
            
        Returns:
            True if successful
        """
        payload = {"content": content}
        return await self._send_webhook(payload)


async def test_discord_bot():
    """Test Discord bot"""
    bot = DiscordBot()
    
    print(f"Discord webhook configured: {bool(bot.webhook_url)}")
    
    if not bot.webhook_url:
        print("Note: Set DISCORD_WEBHOOK_URL in .env to test actual sending")
        return
    
    # Test sending a simple message
    # success = await bot.send_text("🤖 Antigravity-Alpha test message")
    # print(f"Test message sent: {success}")
    
    await bot.close()


if __name__ == "__main__":
    asyncio.run(test_discord_bot())

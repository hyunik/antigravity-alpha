#!/usr/bin/env python3
"""
Weekly Performance Reporter
Generates and sends weekly performance reports to Discord
"""

import asyncio
from datetime import datetime
from typing import Optional
from loguru import logger

from config.settings import settings
from pipeline.data_collector import DataCollector
from analysis.performance_tracker import PerformanceTracker, format_performance_report
from distribution.discord_bot import DiscordBot


class WeeklyReporter:
    """
    Weekly Performance Reporter
    
    Generates comprehensive performance reports and sends them to Discord.
    Designed to run every Monday via GitHub Actions.
    """
    
    def __init__(self):
        self.tracker = PerformanceTracker()
        self.discord = DiscordBot()
        self.collector = DataCollector()
    
    async def update_all_prices(self) -> None:
        """Update prices for all tracked recommendations"""
        await self.collector.initialize()
        
        # Get unique symbols from recommendations
        symbols = list(set(r.symbol for r in self.tracker.recommendations if r.status == "OPEN"))
        
        if not symbols:
            logger.info("No open recommendations to update")
            return
        
        logger.info(f"Updating prices for {len(symbols)} symbols...")
        
        price_map = {}
        for symbol in symbols:
            try:
                # Get OHLCV data for the past week
                df = await self.collector.binance_client.get_ohlcv_df(symbol, "1d", 7)
                if df is not None and len(df) > 0:
                    price_map[symbol] = {
                        "current": df["close"].iloc[-1],
                        "high": df["high"].max(),
                        "low": df["low"].min()
                    }
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")
        
        # Update tracker
        self.tracker.update_prices(price_map)
        logger.info(f"Updated prices for {len(price_map)} symbols")
        
        await self.collector.close()
    
    async def generate_weekly_report(self, days: int = 30) -> str:
        """Generate weekly performance report"""
        self.tracker.load_recommendations()
        
        # Update prices first
        await self.update_all_prices()
        
        # Expire old recommendations
        expired = self.tracker.expire_old_recommendations(days=days)
        if expired > 0:
            logger.info(f"Expired {expired} old recommendations")
        
        # Calculate stats
        stats = self.tracker.calculate_stats(days=days)
        recs = self.tracker.get_recent_recommendations(15)
        
        # Generate report
        report = format_performance_report(stats, recs, days)
        
        return report
    
    async def send_to_discord(self, report: str) -> bool:
        """Send report to Discord"""
        if not settings.discord.webhook_url:
            logger.error("Discord webhook URL not configured")
            return False
        
        # Discord has a 2000 character limit per message
        # Split report into chunks if needed
        header = f"📊 **주간 성과 보고서** ({datetime.utcnow().strftime('%Y-%m-%d')})\n"
        
        # Send as multiple messages if needed
        chunks = self._split_report(report, 1900)
        
        for i, chunk in enumerate(chunks):
            content = f"{header if i == 0 else ''}\n```\n{chunk}\n```"
            success = await self.discord.send_text(content)
            if not success:
                logger.error(f"Failed to send chunk {i+1}")
                return False
            await asyncio.sleep(1)  # Rate limiting
        
        logger.info("Weekly report sent to Discord")
        return True
    
    def _split_report(self, text: str, max_length: int) -> list:
        """Split text into chunks for Discord"""
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += ('\n' if current_chunk else '') + line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def run(self) -> bool:
        """Run weekly report generation and send"""
        logger.info("=" * 50)
        logger.info("Starting Weekly Performance Report")
        logger.info("=" * 50)
        
        try:
            # Generate report
            report = await self.generate_weekly_report(days=30)
            
            # Print to console
            print(report)
            
            # Send to Discord
            success = await self.send_to_discord(report)
            
            await self.discord.close()
            
            if success:
                logger.info("Weekly report completed successfully")
            else:
                logger.warning("Weekly report completed but Discord send failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Weekly report failed: {e}")
            return False


async def main():
    """Main entry point for weekly reporter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Weekly Performance Reporter")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord send")
    
    args = parser.parse_args()
    
    reporter = WeeklyReporter()
    
    if args.no_discord:
        report = await reporter.generate_weekly_report(days=args.days)
        print(report)
    else:
        await reporter.run()


if __name__ == "__main__":
    asyncio.run(main())

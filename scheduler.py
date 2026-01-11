#!/usr/bin/env python3
"""
Antigravity-Alpha Scheduler
Runs analysis periodically on schedule
"""

import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
import sys

from config.settings import settings
from main import AntigravityAlpha


class AntigravityScheduler:
    """
    Scheduler for Antigravity-Alpha analysis
    
    Schedule:
    - Hourly: Quick analysis of 4h data
    - Daily: Full analysis with 1D/1W data
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.alpha = AntigravityAlpha()
        self._is_running = False
    
    async def hourly_analysis(self):
        """Run hourly analysis (quick scan of top coins)"""
        if self._is_running:
            logger.warning("Previous analysis still running, skipping")
            return
        
        self._is_running = True
        logger.info("=" * 50)
        logger.info("Starting HOURLY analysis")
        logger.info("=" * 50)
        
        try:
            await self.alpha.run_analysis(
                coin_limit=100,  # Analyze top 100 for hourly
                min_score=70,    # Higher threshold for hourly
                max_recommendations=20,
                send_to_discord=True,
                use_llm=False    # Skip LLM to save API costs
            )
        except Exception as e:
            logger.error(f"Hourly analysis failed: {e}")
            # Send error notification
            if settings.discord.webhook_url:
                await self.alpha.discord_bot.send_alert(
                    "Hourly Analysis Failed",
                    f"Error: {str(e)[:500]}",
                    color=0xFF0000
                )
        finally:
            self._is_running = False
    
    async def daily_analysis(self):
        """Run daily comprehensive analysis"""
        if self._is_running:
            logger.warning("Previous analysis still running, skipping")
            return
        
        self._is_running = True
        logger.info("=" * 50)
        logger.info("Starting DAILY comprehensive analysis")
        logger.info("=" * 50)
        
        try:
            await self.alpha.run_analysis(
                coin_limit=200,  # Full 200 coins
                min_score=60,    # Standard threshold
                max_recommendations=50,
                send_to_discord=True,
                use_llm=True     # Use LLM for daily analysis
            )
        except Exception as e:
            logger.error(f"Daily analysis failed: {e}")
            if settings.discord.webhook_url:
                await self.alpha.discord_bot.send_alert(
                    "Daily Analysis Failed",
                    f"Error: {str(e)[:500]}",
                    color=0xFF0000
                )
        finally:
            self._is_running = False
    
    def setup_schedules(self):
        """Setup scheduled jobs"""
        # Hourly analysis at minute 0
        self.scheduler.add_job(
            self.hourly_analysis,
            CronTrigger(minute=0),  # Every hour at :00
            id="hourly_analysis",
            name="Hourly Quick Analysis",
            replace_existing=True
        )
        
        # Daily analysis at 00:00 UTC
        self.scheduler.add_job(
            self.daily_analysis,
            CronTrigger(hour=0, minute=0),  # Daily at midnight UTC
            id="daily_analysis",
            name="Daily Comprehensive Analysis",
            replace_existing=True
        )
        
        logger.info("Scheduled jobs configured:")
        logger.info("  - Hourly analysis: Every hour at :00")
        logger.info("  - Daily analysis: 00:00 UTC")
    
    async def run_now(self, mode: str = "daily"):
        """Run analysis immediately"""
        if mode == "hourly":
            await self.hourly_analysis()
        else:
            await self.daily_analysis()
    
    async def start(self, run_immediately: bool = False):
        """Start the scheduler"""
        logger.info("Starting Antigravity-Alpha Scheduler")
        
        # Validate settings
        errors = settings.validate()
        if errors:
            for error in errors:
                logger.warning(f"Config warning: {error}")
        
        # Setup schedules
        self.setup_schedules()
        
        # Start scheduler
        self.scheduler.start()
        logger.info("Scheduler started")
        
        # Send startup notification
        if settings.discord.webhook_url:
            await self.alpha.discord_bot.send_text(
                "🚀 **Antigravity-Alpha Scheduler Started**\n"
                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                "Schedules:\n"
                "• Hourly analysis: Every hour at :00\n"
                "• Daily analysis: 00:00 UTC"
            )
        
        # Run immediately if requested
        if run_immediately:
            logger.info("Running initial analysis...")
            await self.hourly_analysis()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler shutting down...")
            self.scheduler.shutdown()
            await self.alpha.close()
    
    async def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        await self.alpha.close()
        logger.info("Scheduler stopped")


async def main():
    """Main entry point for scheduler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Antigravity-Alpha Scheduler")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run analysis immediately on startup"
    )
    parser.add_argument(
        "--once",
        choices=["hourly", "daily"],
        help="Run single analysis and exit"
    )
    
    args = parser.parse_args()
    
    scheduler = AntigravityScheduler()
    
    if args.once:
        await scheduler.run_now(args.once)
        await scheduler.alpha.close()
    else:
        await scheduler.start(run_immediately=args.run_now)


if __name__ == "__main__":
    asyncio.run(main())

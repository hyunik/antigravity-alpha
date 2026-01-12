#!/usr/bin/env python3
"""
Antigravity-Alpha Main Entry Point
Smart Coin Selection and Trading Strategy System
"""

import asyncio
import argparse
from datetime import datetime
from typing import List, Optional
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/antigravity_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)

from config.settings import settings
from data.models import init_database
from pipeline.data_collector import DataCollector
from analysis.scoring_engine import ScoringEngine, CoinScore
from agents.cio_agent import CIOAgent, TradeRecommendation
from distribution.discord_bot import DiscordBot


class AntigravityAlpha:
    """
    Main orchestrator for Antigravity-Alpha system
    
    Workflow:
    1. [Data Agent] Collect data for top 200 coins
    2. [Logic Agent] Detect patterns and score coins
    3. [CIO Agent] LLM-powered final analysis
    4. [Report Agent] Send to Discord
    """
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.scoring_engine = ScoringEngine()
        self.cio_agent = CIOAgent()
        self.discord_bot = DiscordBot()
        
        # Initialize database
        self.engine = init_database(settings.system.database_url)
        logger.info("Antigravity-Alpha initialized")
    
    async def close(self):
        """Cleanup resources"""
        await self.data_collector.close()
        await self.discord_bot.close()
    
    async def run_analysis(
        self,
        coin_limit: int = 200,
        min_score: float = 60,
        max_recommendations: int = 50,
        send_to_discord: bool = True,
        use_llm: bool = True
    ) -> List[TradeRecommendation]:
        """
        Run full analysis pipeline
        
        Args:
            coin_limit: Number of top coins to analyze
            min_score: Minimum score for recommendations
            max_recommendations: Maximum recommendations to generate
            send_to_discord: Whether to send results to Discord
            use_llm: Whether to use LLM for final analysis
            
        Returns:
            List of TradeRecommendation
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting analysis run at {start_time}")
        
        try:
            # Step 1: Initialize and collect data
            logger.info("Step 1: Initializing data collector...")
            await self.data_collector.initialize()
            
            # Check if in CoinGecko-only mode (exchanges blocked)
            is_coingecko_only = getattr(self.data_collector, '_coingecko_only', False)
            if is_coingecko_only:
                logger.warning("Running in CoinGecko-only mode (exchanges blocked)")
            
            logger.info(f"Step 2: Collecting data for top {coin_limit} coins...")
            coins, ohlcv_data = await self.data_collector.collect_all_data(
                limit=coin_limit,
                timeframes=["4h", "1d"],
                use_binance_direct=not is_coingecko_only  # Use Binance directly when available
            )
            logger.info(f"Collected data for {len(coins)} coins")
            
            # Step 2: Score all coins
            logger.info("Step 3: Scoring coins...")
            coin_scores: List[CoinScore] = []
            
            for coin in coins:
                symbol = coin["binance_symbol"]
                if symbol not in ohlcv_data:
                    continue
                
                df_4h = ohlcv_data[symbol].get("4h")
                df_1d = ohlcv_data[symbol].get("1d")
                
                if df_4h is None or len(df_4h) < 30:
                    logger.debug(f"Skipping {symbol} - insufficient data")
                    continue
                
                try:
                    # Get market data (skip in CoinGecko-only mode)
                    if is_coingecko_only:
                        market_data = {"open_interest": 0, "funding_rate": 0}
                    else:
                        market_data = await self.data_collector.fetch_market_data(symbol)
                    
                    score = self.scoring_engine.score_coin(
                        symbol=symbol,
                        df_4h=df_4h,
                        df_1d=df_1d,
                        open_interest=market_data.get("open_interest", 0),
                        funding_rate=market_data.get("funding_rate", 0)
                    )
                    coin_scores.append(score)
                    
                    if score.total_score >= min_score:
                        logger.info(f"  {symbol}: Score {score.total_score:.1f} | {score.direction} | {score.conviction}")
                    
                except Exception as e:
                    logger.error(f"Error scoring {symbol}: {e}")
            
            logger.info(f"Scored {len(coin_scores)} coins")
            
            # Step 3: Filter and rank
            qualified = self.scoring_engine.rank_coins(coin_scores, min_score)
            logger.info(f"Step 4: {len(qualified)} coins meet minimum score of {min_score}")
            
            # Step 4: LLM Analysis (optional)
            recommendations: List[TradeRecommendation] = []
            price_map = {coin["binance_symbol"]: coin["current_price"] for coin in coins}
            
            if use_llm and settings.llm.openai_api_key or settings.llm.gemini_api_key:
                logger.info("Step 5: Running CIO Agent analysis...")
                recommendations = await self.cio_agent.batch_analyze(
                    qualified[:max_recommendations],
                    price_map,
                    min_score=min_score,
                    max_recommendations=max_recommendations
                )
                logger.info(f"CIO Agent produced {len(recommendations)} recommendations")
            else:
                # Fallback: Create recommendations without LLM
                logger.info("Step 5: Creating recommendations without LLM...")
                for score in qualified[:max_recommendations]:
                    price = price_map.get(score.symbol, 0)
                    if price > 0:
                        rec = self.cio_agent._create_fallback_recommendation(score, price)
                        recommendations.append(rec)
            
            # Step 5: Send to Discord
            if send_to_discord:
                logger.info("Step 6: Sending to Discord...")
                if settings.discord.webhook_url:
                    if recommendations:
                        results = await self.discord_bot.send_batch(
                            recommendations,
                            total_analyzed=len(coins),
                            include_summary=True,
                            max_recommendations=10
                        )
                        logger.info(f"Discord: {results['success']} sent, {results['failed']} failed")
                    else:
                        # Send notification even when no recommendations
                        # Include list of qualified coins for reference
                        qualified_list = ""
                        if qualified:
                            qualified_names = [f"{s.symbol}({s.total_score:.0f}점/{s.direction})" for s in qualified[:10]]
                            qualified_list = f"\n\n📋 **참고 - 기준 충족 코인:**\n" + ", ".join(qualified_names)
                        
                        await self.discord_bot.send_text(
                            f"📊 분석 완료: {len(coins)}개 코인 분석, {len(qualified)}개 기준 충족\n"
                            f"⚠️ CIO 최종 추천 없음 (LLM이 진입 부적합 판단)\n"
                            f"{'🌐 CoinGecko 모드 (거래소 차단됨)' if is_coingecko_only else ''}"
                            f"{qualified_list}"
                        )
                        logger.info("Discord: Sent no-recommendation status with qualified list")
                else:
                    logger.warning("Discord webhook not configured, skipping send")
            
            # Log summary
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("=" * 50)
            logger.info("Analysis Complete")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Coins analyzed: {len(coins)}")
            logger.info(f"Recommendations: {len(recommendations)}")
            logger.info("=" * 50)
            
            # Print top recommendations
            for i, rec in enumerate(recommendations[:5]):
                logger.info(
                    f"  {i+1}. {rec.symbol} | {rec.direction} | "
                    f"Score: {rec.score:.1f} | Entry: ${rec.entry_zone[0]:.4f}-${rec.entry_zone[1]:.4f}"
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Send error to Discord
            if send_to_discord and settings.discord.webhook_url:
                try:
                    await self.discord_bot.send_text(
                        f"❌ 분석 실패: {str(e)[:200]}"
                    )
                except:
                    pass
            raise
    
    async def run_quick_scan(self, symbol: str) -> Optional[CoinScore]:
        """
        Quick scan for a single coin
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            
        Returns:
            CoinScore or None
        """
        logger.info(f"Quick scan for {symbol}")
        
        await self.data_collector.initialize()
        
        # Fetch data
        ohlcv = await self.data_collector.fetch_multi_timeframe_data(symbol, ["4h", "1d"])
        market_data = await self.data_collector.fetch_market_data(symbol)
        
        if "4h" not in ohlcv:
            logger.error(f"No 4h data for {symbol}")
            return None
        
        # Score
        score = self.scoring_engine.score_coin(
            symbol=symbol,
            df_4h=ohlcv["4h"],
            df_1d=ohlcv.get("1d"),
            open_interest=market_data.get("open_interest", 0),
            funding_rate=market_data.get("funding_rate", 0)
        )
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Quick Scan: {symbol}")
        print(f"{'='*50}")
        print(f"Total Score: {score.total_score:.1f}/100")
        print(f"  ICT Score: {score.ict_score:.1f}")
        print(f"  Wyckoff Score: {score.wyckoff_score:.1f}")
        print(f"  VCP Score: {score.vcp_score:.1f}")
        print(f"  Market Score: {score.market_score:.1f}")
        print(f"Direction: {score.direction}")
        print(f"Conviction: {score.conviction}")
        print(f"Wyckoff Phase: {score.wyckoff_phase}")
        print(f"\nPatterns Detected:")
        print(f"  MSS: {score.has_mss}")
        print(f"  FVG: {score.has_fvg}")
        print(f"  Spring: {score.has_spring}")
        print(f"  VCP: {score.has_vcp}")
        print(f"\nTrade Setup:")
        print(f"  Entry: ${score.entry_zone[0]:.4f} - ${score.entry_zone[1]:.4f}")
        print(f"  Stop Loss: ${score.stop_loss:.4f}")
        print(f"  Targets: {[f'${t:.4f}' for t in score.targets]}")
        print(f"\nReasons:")
        for r in score.reasons:
            print(f"  • {r}")
        print(f"\nRisk Factors:")
        for r in score.risk_factors:
            print(f"  ⚠️ {r}")
        
        return score


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Antigravity-Alpha Smart Coin Selection")
    parser.add_argument(
        "--mode",
        choices=["full", "quick"],
        default="full",
        help="Analysis mode: full (200 coins) or quick (single coin)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol for quick scan mode"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of coins to analyze in full mode"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=60,
        help="Minimum score for recommendations"
    )
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="Skip Discord sending"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM analysis"
    )
    
    args = parser.parse_args()
    
    # Validate settings
    errors = settings.validate()
    if errors:
        for error in errors:
            logger.warning(f"Config warning: {error}")
    
    alpha = AntigravityAlpha()
    
    try:
        if args.mode == "quick":
            await alpha.run_quick_scan(args.symbol)
        else:
            await alpha.run_analysis(
                coin_limit=args.limit,
                min_score=args.min_score,
                send_to_discord=not args.no_discord,
                use_llm=not args.no_llm
            )
    finally:
        await alpha.close()


if __name__ == "__main__":
    asyncio.run(main())

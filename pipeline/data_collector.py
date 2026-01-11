"""
Unified Data Collector
Orchestrates data collection from multiple sources with failover support
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from loguru import logger
import pandas as pd

from .coingecko_client import CoinGeckoClient
from .binance_client import BinanceClient
from .bybit_client import BybitClient


class DataCollector:
    """
    Unified data collector that orchestrates data fetching from multiple sources.
    Implements failover from Binance to Bybit.
    """
    
    TIMEFRAMES = ["1h", "4h", "1d", "1w"]
    BATCH_SIZE = 20  # Process 20 coins at a time to avoid rate limits
    
    def __init__(self):
        self.coingecko = CoinGeckoClient(rate_limit_delay=1.5)
        self.binance = BinanceClient(rate_limit_delay=0.1)
        self.bybit = BybitClient(rate_limit_delay=0.1)
        
        # Cache for available trading pairs
        self._binance_pairs: Optional[List[str]] = None
        self._symbol_mapping: Dict[str, str] = {}  # CoinGecko symbol -> Binance symbol
    
    async def close(self):
        """Close all client sessions"""
        await self.coingecko.close()
        await self.binance.close()
        await self.bybit.close()
    
    async def initialize(self) -> bool:
        """
        Initialize the collector by fetching available trading pairs
        Tries Binance first, falls back to Bybit if blocked
        
        Returns:
            True if initialization successful
        """
        try:
            # Try Binance first
            self._binance_pairs = await self.binance.get_usdt_pairs()
            if self._binance_pairs and len(self._binance_pairs) > 0:
                logger.info(f"Initialized with {len(self._binance_pairs)} Binance USDT pairs")
                self._use_bybit_primary = False
                return True
        except Exception as e:
            logger.warning(f"Binance initialization failed (may be region blocked): {e}")
        
        # Fallback to Bybit
        try:
            bybit_pairs = await self.bybit.get_usdt_pairs()
            if bybit_pairs and len(bybit_pairs) > 0:
                self._binance_pairs = bybit_pairs  # Use same format
                self._use_bybit_primary = True
                logger.info(f"Using Bybit as primary source with {len(self._binance_pairs)} pairs")
                return True
        except Exception as e:
            logger.error(f"Bybit initialization also failed: {e}")
        
        logger.error("Failed to initialize any exchange")
        return False
    
    def _get_binance_symbol(self, coingecko_symbol: str) -> Optional[str]:
        """Convert CoinGecko symbol to Binance trading pair"""
        binance_symbol = f"{coingecko_symbol.upper()}USDT"
        if self._binance_pairs and binance_symbol in self._binance_pairs:
            return binance_symbol
        return None
    
    async def get_top_coins(self, limit: int = 200) -> List[Dict]:
        """
        Get top coins by market cap from CoinGecko
        
        Args:
            limit: Number of coins to fetch
            
        Returns:
            List of coin data with Binance symbol mapping
        """
        raw_coins = await self.coingecko.get_top_coins(limit)
        coins = self.coingecko.parse_top_coins(raw_coins)
        
        # Add Binance symbol mapping
        tradeable_coins = []
        for coin in coins:
            binance_symbol = self._get_binance_symbol(coin["symbol"])
            if binance_symbol:
                coin["binance_symbol"] = binance_symbol
                tradeable_coins.append(coin)
            else:
                logger.debug(f"No Binance pair for {coin['symbol']}")
        
        logger.info(f"Found {len(tradeable_coins)}/{len(coins)} tradeable coins on Binance Futures")
        return tradeable_coins
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 500,
        use_bybit_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with fallback support
        
        Args:
            symbol: Binance trading pair symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (1h, 4h, 1d, 1w)
            limit: Number of candles to fetch
            use_bybit_fallback: Whether to use Bybit as fallback
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try Binance first
        try:
            df = await self.binance.get_ohlcv_df(symbol, timeframe, limit)
            if not df.empty:
                df["source"] = "binance"
                return df
        except Exception as e:
            logger.warning(f"Binance OHLCV failed for {symbol}: {e}")
        
        # Fallback to Bybit
        if use_bybit_fallback:
            try:
                df = await self.bybit.get_ohlcv_df(symbol, timeframe, limit)
                if not df.empty:
                    df["source"] = "bybit"
                    logger.info(f"Using Bybit fallback for {symbol}")
                    return df
            except Exception as e:
                logger.warning(f"Bybit OHLCV failed for {symbol}: {e}")
        
        return pd.DataFrame()
    
    async def fetch_market_data(self, symbol: str) -> Dict:
        """
        Fetch market data (OI, Funding Rate) with fallback
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with market data
        """
        # Try Binance first
        try:
            data = await self.binance.get_market_data(symbol)
            if data.get("open_interest", 0) > 0:
                data["source"] = "binance"
                return data
        except Exception as e:
            logger.warning(f"Binance market data failed for {symbol}: {e}")
        
        # Fallback to Bybit
        try:
            oi = await self.bybit.get_open_interest(symbol)
            funding = await self.bybit.get_funding_rate(symbol)
            
            return {
                "symbol": symbol,
                "open_interest": float(oi.get("openInterest", 0)) if oi else 0,
                "funding_rate": float(funding.get("fundingRate", 0)) if funding else 0,
                "source": "bybit"
            }
        except Exception as e:
            logger.warning(f"Bybit market data failed for {symbol}: {e}")
        
        return {"symbol": symbol, "open_interest": 0, "funding_rate": 0, "source": None}
    
    async def fetch_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple timeframes
        
        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframes to fetch (default: all)
            limit: Number of candles per timeframe
            
        Returns:
            Dict mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = self.TIMEFRAMES
        
        results = {}
        for tf in timeframes:
            df = await self.fetch_ohlcv(symbol, tf, limit)
            if not df.empty:
                results[tf] = df
            else:
                logger.warning(f"No data for {symbol} {tf}")
        
        return results
    
    async def fetch_batch(
        self,
        symbols: List[str],
        timeframe: str = "4h",
        include_market_data: bool = True
    ) -> List[Dict]:
        """
        Fetch data for a batch of symbols (parallel processing)
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe to fetch
            include_market_data: Whether to include OI/funding data
            
        Returns:
            List of data dictionaries for each symbol
        """
        async def fetch_single(symbol: str) -> Dict:
            result = {"symbol": symbol}
            
            # Fetch OHLCV
            ohlcv = await self.fetch_ohlcv(symbol, timeframe)
            result["ohlcv"] = ohlcv
            
            # Fetch market data if requested
            if include_market_data:
                market = await self.fetch_market_data(symbol)
                result.update(market)
            
            return result
        
        # Process in batches to avoid rate limits
        all_results = []
        for i in range(0, len(symbols), self.BATCH_SIZE):
            batch = symbols[i:i + self.BATCH_SIZE]
            logger.info(f"Processing batch {i//self.BATCH_SIZE + 1}/{(len(symbols) + self.BATCH_SIZE - 1)//self.BATCH_SIZE}")
            
            tasks = [fetch_single(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch fetch error: {result}")
                else:
                    all_results.append(result)
            
            # Small delay between batches
            if i + self.BATCH_SIZE < len(symbols):
                await asyncio.sleep(1)
        
        return all_results
    
    async def collect_all_data(
        self,
        limit: int = 200,
        timeframes: Optional[List[str]] = None
    ) -> Tuple[List[Dict], Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Collect all data for top coins
        
        Args:
            limit: Number of top coins to collect
            timeframes: Timeframes to collect (default: 4h, 1d)
            
        Returns:
            Tuple of (coins_data, ohlcv_data)
            - coins_data: List of coin metadata
            - ohlcv_data: Dict[symbol -> Dict[timeframe -> DataFrame]]
        """
        if timeframes is None:
            timeframes = ["4h", "1d"]
        
        # Get top coins
        logger.info(f"Fetching top {limit} coins from CoinGecko...")
        coins = await self.get_top_coins(limit)
        
        # Fetch OHLCV for each coin
        logger.info(f"Fetching OHLCV data for {len(coins)} coins...")
        ohlcv_data = {}
        
        for i, coin in enumerate(coins):
            symbol = coin["binance_symbol"]
            logger.info(f"[{i+1}/{len(coins)}] Fetching {symbol}...")
            
            ohlcv_data[symbol] = await self.fetch_multi_timeframe_data(
                symbol, timeframes
            )
            
            # Rate limiting
            if (i + 1) % 10 == 0:
                await asyncio.sleep(1)
        
        return coins, ohlcv_data


async def test_collector():
    """Test DataCollector"""
    collector = DataCollector()
    try:
        # Initialize
        await collector.initialize()
        
        # Get top coins
        coins = await collector.get_top_coins(10)
        print(f"\nTop {len(coins)} tradeable coins:")
        for coin in coins[:5]:
            print(f"  {coin['market_cap_rank']}. {coin['symbol']} -> {coin['binance_symbol']}")
        
        # Fetch multi-timeframe data for BTC
        print("\nFetching BTC multi-timeframe data...")
        data = await collector.fetch_multi_timeframe_data("BTCUSDT", ["1h", "4h"])
        for tf, df in data.items():
            print(f"  {tf}: {len(df)} candles")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(test_collector())

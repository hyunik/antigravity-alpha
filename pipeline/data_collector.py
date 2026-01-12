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
        Tries exchanges first, falls back to CoinGecko-only mode if all fail
        
        Returns:
            True if initialization successful
        """
        self._use_bybit_primary = False
        self._coingecko_only = False
        
        # Try Binance first
        try:
            self._binance_pairs = await self.binance.get_usdt_pairs()
            if self._binance_pairs and len(self._binance_pairs) > 0:
                logger.info(f"Initialized with {len(self._binance_pairs)} Binance USDT pairs")
                return True
        except Exception as e:
            logger.warning(f"Binance initialization failed (may be region blocked): {e}")
        
        # Fallback to Bybit
        try:
            bybit_pairs = await self.bybit.get_usdt_pairs()
            if bybit_pairs and len(bybit_pairs) > 0:
                self._binance_pairs = bybit_pairs
                self._use_bybit_primary = True
                logger.info(f"Using Bybit as primary source with {len(self._binance_pairs)} pairs")
                return True
        except Exception as e:
            logger.warning(f"Bybit initialization also failed: {e}")
        
        # Final fallback: CoinGecko only mode
        logger.warning("All exchanges blocked. Using CoinGecko-only mode.")
        self._coingecko_only = True
        self._binance_pairs = []  # Will use CoinGecko IDs instead
        return True  # Still return True to continue with CoinGecko
    
    def _get_binance_symbol(self, coingecko_symbol: str) -> Optional[str]:
        """Convert CoinGecko symbol to Binance trading pair"""
        if self._coingecko_only:
            # In CoinGecko-only mode, return symbol as-is
            return f"{coingecko_symbol.upper()}USDT"
        
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
            List of coin data with trading symbol mapping
        """
        raw_coins = await self.coingecko.get_top_coins(limit)
        coins = self.coingecko.parse_top_coins(raw_coins)
        
        # Add CoinGecko ID for OHLC fetching
        for i, coin in enumerate(coins):
            coin["coingecko_id"] = raw_coins[i].get("id") if i < len(raw_coins) else None
        
        if self._coingecko_only:
            # In CoinGecko-only mode, use all coins
            tradeable_coins = []
            for coin in coins:
                coin["binance_symbol"] = f"{coin['symbol'].upper()}USDT"
                tradeable_coins.append(coin)
            logger.info(f"CoinGecko-only mode: Using {len(tradeable_coins)}/{len(coins)} coins")
            return tradeable_coins[:50]  # Limit to 50 to avoid rate limits
        
        # Normal mode: Filter for tradeable pairs
        tradeable_coins = []
        for coin in coins:
            binance_symbol = self._get_binance_symbol(coin["symbol"])
            if binance_symbol:
                coin["binance_symbol"] = binance_symbol
                tradeable_coins.append(coin)
            else:
                logger.debug(f"No Binance pair for {coin['symbol']}")
        
        logger.info(f"Found {len(tradeable_coins)}/{len(coins)} tradeable coins")
        return tradeable_coins
    
    async def get_binance_coins(self, limit: int = 200) -> List[Dict]:
        """
        Get coins directly from Binance USDT perpetual pairs
        
        Args:
            limit: Maximum number of coins to return
            
        Returns:
            List of coin data with Binance symbols
        """
        if not self._binance_pairs:
            logger.warning("No Binance pairs available, falling back to CoinGecko")
            return await self.get_top_coins(limit)
        
        # Sort by symbol (you could also get volume data to sort by)
        # For now, just use all available pairs
        coins = []
        for symbol in list(self._binance_pairs)[:limit]:
            if symbol.endswith("USDT"):
                base_symbol = symbol.replace("USDT", "")
                coins.append({
                    "symbol": base_symbol,
                    "binance_symbol": symbol,
                    "coingecko_id": None,  # Not needed for Binance mode
                    "name": base_symbol,
                    "market_cap_rank": None,
                    "current_price": 0,  # Will be fetched from OHLCV
                })
        
        logger.info(f"Using {len(coins)} Binance USDT pairs directly")
        return coins
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 500,
        use_bybit_fallback: bool = True,
        coingecko_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with fallback support
        
        Args:
            symbol: Binance trading pair symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (1h, 4h, 1d, 1w)
            limit: Number of candles to fetch
            use_bybit_fallback: Whether to use Bybit as fallback
            coingecko_id: CoinGecko ID for CoinGecko-only mode
            
        Returns:
            DataFrame with OHLCV data
        """
        # CoinGecko-only mode
        if self._coingecko_only and coingecko_id:
            try:
                # CoinGecko OHLC only supports certain day ranges
                days_map = {"1h": 1, "4h": 7, "1d": 30, "1w": 90}
                days = days_map.get(timeframe, 30)
                
                ohlc = await self.coingecko.get_coin_ohlc(coingecko_id, days)
                if ohlc:
                    df = pd.DataFrame(ohlc, columns=["timestamp", "open", "high", "low", "close"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df["volume"] = 0  # CoinGecko OHLC doesn't have volume
                    df["source"] = "coingecko"
                    return df
            except Exception as e:
                logger.warning(f"CoinGecko OHLC failed for {coingecko_id}: {e}")
            return pd.DataFrame()
        
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
        limit: int = 500,
        coingecko_id: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple timeframes
        
        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframes to fetch (default: all)
            limit: Number of candles per timeframe
            coingecko_id: CoinGecko ID for CoinGecko-only mode
            
        Returns:
            Dict mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = self.TIMEFRAMES
        
        results = {}
        for tf in timeframes:
            df = await self.fetch_ohlcv(symbol, tf, limit, coingecko_id=coingecko_id)
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
        timeframes: Optional[List[str]] = None,
        use_binance_direct: bool = False
    ) -> Tuple[List[Dict], Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Collect all data for top coins
        
        Args:
            limit: Number of top coins to collect
            timeframes: Timeframes to collect (default: 4h, 1d)
            use_binance_direct: If True, use Binance pairs directly instead of CoinGecko
            
        Returns:
            Tuple of (coins_data, ohlcv_data)
            - coins_data: List of coin metadata
            - ohlcv_data: Dict[symbol -> Dict[timeframe -> DataFrame]]
        """
        if timeframes is None:
            timeframes = ["4h", "1d"]
        
        # Get coins based on mode
        if use_binance_direct and self._binance_pairs and not self._coingecko_only:
            logger.info(f"Fetching {limit} coins from Binance USDT pairs directly...")
            coins = await self.get_binance_coins(limit)
        else:
            logger.info(f"Fetching top {limit} coins from CoinGecko...")
            coins = await self.get_top_coins(limit)
        
        # Fetch OHLCV for each coin
        logger.info(f"Fetching OHLCV data for {len(coins)} coins...")
        ohlcv_data = {}
        
        for i, coin in enumerate(coins):
            symbol = coin["binance_symbol"]
            coingecko_id = coin.get("coingecko_id")
            logger.info(f"[{i+1}/{len(coins)}] Fetching {symbol}...")
            
            ohlcv_data[symbol] = await self.fetch_multi_timeframe_data(
                symbol, timeframes, coingecko_id=coingecko_id
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

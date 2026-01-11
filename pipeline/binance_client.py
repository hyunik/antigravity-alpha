"""
Binance API Client
For fetching detailed OHLCV data and futures market data (OI, Funding Rate)
Uses public endpoints - no API key required for most data
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd


class BinanceClient:
    """Binance API client for OHLCV and futures data"""
    
    SPOT_BASE_URL = "https://api.binance.com/api/v3"
    FUTURES_BASE_URL = "https://fapi.binance.com/fapi/v1"
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1h": "1h",
        "4h": "4h", 
        "1d": "1d",
        "1w": "1w"
    }
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """
        Initialize Binance client
        
        Args:
            rate_limit_delay: Delay between requests in seconds
        """
        self.rate_limit_delay = rate_limit_delay
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
    
    async def _request(self, base_url: str, endpoint: str, params: Optional[Dict] = None) -> Optional[any]:
        """Make API request with rate limiting"""
        session = await self._get_session()
        url = f"{base_url}{endpoint}"
        
        try:
            await asyncio.sleep(self.rate_limit_delay)
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("Binance rate limit hit, waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self._request(base_url, endpoint, params)
                elif response.status == 418:
                    logger.error("Binance IP banned! Waiting 5 minutes...")
                    await asyncio.sleep(300)
                    return None
                else:
                    text = await response.text()
                    logger.error(f"Binance API error {response.status}: {text}")
                    return None
        except Exception as e:
            logger.error(f"Binance request failed: {e}")
            return None
    
    async def get_exchange_info(self) -> Optional[Dict]:
        """Get exchange information including all trading pairs"""
        return await self._request(self.SPOT_BASE_URL, "/exchangeInfo")
    
    async def get_futures_exchange_info(self) -> Optional[Dict]:
        """Get futures exchange information"""
        return await self._request(self.FUTURES_BASE_URL, "/exchangeInfo")
    
    async def get_usdt_pairs(self) -> List[str]:
        """Get all USDT trading pairs from futures"""
        info = await self.get_futures_exchange_info()
        if not info:
            return []
        
        pairs = []
        for symbol in info.get("symbols", []):
            if symbol.get("quoteAsset") == "USDT" and symbol.get("status") == "TRADING":
                pairs.append(symbol["symbol"])
        
        return pairs
    
    async def get_klines(
        self, 
        symbol: str, 
        interval: str = "1h", 
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        Get kline/candlestick data
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)
            limit: Number of klines (max 1500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of klines [open_time, open, high, low, close, volume, ...]
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        return await self._request(self.SPOT_BASE_URL, "/klines", params) or []
    
    async def get_futures_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500
    ) -> List[List]:
        """Get futures kline data"""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return await self._request(self.FUTURES_BASE_URL, "/klines", params) or []
    
    async def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """Get current open interest for a symbol"""
        return await self._request(self.FUTURES_BASE_URL, "/openInterest", {"symbol": symbol})
    
    async def get_funding_rate(self, symbol: str) -> Optional[List]:
        """Get recent funding rates for a symbol"""
        return await self._request(self.FUTURES_BASE_URL, "/fundingRate", {
            "symbol": symbol,
            "limit": 10
        })
    
    async def get_long_short_ratio(self, symbol: str) -> Optional[List]:
        """Get top trader long/short ratio"""
        return await self._request(
            self.FUTURES_BASE_URL.replace("/fapi/v1", "/futures/data"),
            "/topLongShortAccountRatio",
            {"symbol": symbol, "period": "1h", "limit": 10}
        )
    
    def parse_klines_to_df(self, klines: List[List]) -> pd.DataFrame:
        """
        Parse klines to pandas DataFrame
        
        Args:
            klines: Raw kline data from Binance
            
        Returns:
            DataFrame with OHLCV columns
        """
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    async def get_ohlcv_df(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get OHLCV data as DataFrame
        
        Args:
            symbol: Trading pair symbol
            interval: Timeframe
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        klines = await self.get_futures_klines(symbol, interval, limit)
        return self.parse_klines_to_df(klines)
    
    async def get_market_data(self, symbol: str) -> Dict:
        """
        Get all market data for a symbol (OI, funding rate)
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with market data
        """
        oi_data = await self.get_open_interest(symbol)
        funding_data = await self.get_funding_rate(symbol)
        
        result = {
            "symbol": symbol,
            "open_interest": float(oi_data.get("openInterest", 0)) if oi_data else 0,
            "funding_rate": float(funding_data[0].get("fundingRate", 0)) if funding_data else 0,
            "next_funding_time": funding_data[0].get("fundingTime") if funding_data else None
        }
        
        return result


async def test_binance():
    """Test Binance client"""
    client = BinanceClient()
    try:
        # Test USDT pairs
        pairs = await client.get_usdt_pairs()
        print(f"Found {len(pairs)} USDT pairs")
        
        # Test OHLCV
        df = await client.get_ohlcv_df("BTCUSDT", "4h", 100)
        print(f"\nBTCUSDT 4h OHLCV (last 5 candles):")
        print(df.tail())
        
        # Test market data
        market_data = await client.get_market_data("BTCUSDT")
        print(f"\nBTCUSDT Market Data:")
        print(f"  Open Interest: {market_data['open_interest']:,.0f}")
        print(f"  Funding Rate: {market_data['funding_rate']:.6f}")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_binance())

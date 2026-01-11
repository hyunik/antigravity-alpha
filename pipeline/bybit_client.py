"""
Bybit API Client (Backup)
Fallback data source when Binance is unavailable
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional
from loguru import logger
import pandas as pd


class BybitClient:
    """Bybit API client as backup data source"""
    
    BASE_URL = "https://api.bybit.com"
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """Initialize Bybit client"""
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
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with rate limiting"""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            await asyncio.sleep(self.rate_limit_delay)
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("retCode") == 0:
                        return data.get("result")
                    else:
                        logger.error(f"Bybit API error: {data.get('retMsg')}")
                        return None
                else:
                    logger.error(f"Bybit HTTP error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Bybit request failed: {e}")
            return None
    
    async def get_tickers(self, category: str = "linear") -> List[Dict]:
        """Get all tickers for a category"""
        result = await self._request("/v5/market/tickers", {"category": category})
        return result.get("list", []) if result else []
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "60",  # 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        limit: int = 200,
        category: str = "linear"
    ) -> List[List]:
        """
        Get kline data
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: 1,3,5,15,30,60,120,240,360,720,D,W,M
            limit: Number of klines (max 1000)
            category: linear (USDT perpetual) / inverse / spot
            
        Returns:
            List of klines
        """
        # Convert interval format
        interval_map = {
            "1h": "60",
            "4h": "240",
            "1d": "D",
            "1w": "W"
        }
        bybit_interval = interval_map.get(interval, interval)
        
        result = await self._request("/v5/market/kline", {
            "category": category,
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": limit
        })
        
        return result.get("list", []) if result else []
    
    async def get_open_interest(self, symbol: str, category: str = "linear") -> Optional[Dict]:
        """Get open interest for a symbol"""
        result = await self._request("/v5/market/open-interest", {
            "category": category,
            "symbol": symbol,
            "intervalTime": "1h",
            "limit": 1
        })
        if result and result.get("list"):
            return result["list"][0]
        return None
    
    async def get_funding_rate(self, symbol: str, category: str = "linear") -> Optional[Dict]:
        """Get funding rate history"""
        result = await self._request("/v5/market/funding/history", {
            "category": category,
            "symbol": symbol,
            "limit": 1
        })
        if result and result.get("list"):
            return result["list"][0]
        return None
    
    def parse_klines_to_df(self, klines: List[List]) -> pd.DataFrame:
        """Parse Bybit klines to DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        # Bybit returns: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        # Sort by timestamp ascending (Bybit returns descending)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    async def get_ohlcv_df(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200
    ) -> pd.DataFrame:
        """Get OHLCV data as DataFrame"""
        klines = await self.get_klines(symbol, interval, limit)
        return self.parse_klines_to_df(klines)


async def test_bybit():
    """Test Bybit client"""
    client = BybitClient()
    try:
        # Test OHLCV
        df = await client.get_ohlcv_df("BTCUSDT", "4h", 100)
        print("BTCUSDT 4h OHLCV (last 5 candles):")
        print(df.tail())
        
        # Test OI
        oi = await client.get_open_interest("BTCUSDT")
        print(f"\nOpen Interest: {oi}")
        
        # Test funding
        funding = await client.get_funding_rate("BTCUSDT")
        print(f"Funding Rate: {funding}")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_bybit())

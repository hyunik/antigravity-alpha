"""
CoinGecko API Client
Free API for fetching top coins by market cap and basic market data
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger


class CoinGeckoClient:
    """CoinGecko API client for free market data"""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, rate_limit_delay: float = 1.5):
        """
        Initialize CoinGecko client
        
        Args:
            rate_limit_delay: Delay between requests in seconds (CoinGecko free tier: ~10-30 req/min)
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
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with rate limiting"""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("CoinGecko rate limit hit, waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self._request(endpoint, params)
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"CoinGecko request failed: {e}")
            return None
    
    async def get_top_coins(self, limit: int = 200) -> List[Dict]:
        """
        Get top coins by market cap
        
        Args:
            limit: Number of coins to fetch (max 250 per page)
            
        Returns:
            List of coin data dictionaries
        """
        coins = []
        per_page = min(limit, 250)
        pages = (limit + per_page - 1) // per_page
        
        for page in range(1, pages + 1):
            data = await self._request("/coins/markets", params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "24h,7d"
            })
            
            if data:
                coins.extend(data)
                logger.info(f"Fetched page {page}/{pages} from CoinGecko ({len(data)} coins)")
            
            if len(coins) >= limit:
                break
        
        return coins[:limit]
    
    async def get_coin_ohlc(self, coin_id: str, days: int = 30) -> List[List]:
        """
        Get OHLC data for a coin
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
            
        Returns:
            List of [timestamp, open, high, low, close]
        """
        data = await self._request(f"/coins/{coin_id}/ohlc", params={
            "vs_currency": "usd",
            "days": days
        })
        return data if data else []
    
    async def get_coin_market_chart(self, coin_id: str, days: int = 30) -> Optional[Dict]:
        """
        Get market chart data (prices, market_caps, total_volumes)
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days
            
        Returns:
            Dict with prices, market_caps, total_volumes
        """
        return await self._request(f"/coins/{coin_id}/market_chart", params={
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        })
    
    def parse_top_coins(self, coins_data: List[Dict]) -> List[Dict]:
        """
        Parse top coins data into simplified format
        
        Args:
            coins_data: Raw coin data from CoinGecko
            
        Returns:
            List of parsed coin dictionaries
        """
        parsed = []
        for coin in coins_data:
            parsed.append({
                "coingecko_id": coin.get("id"),
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "current_price": coin.get("current_price"),
                "market_cap": coin.get("market_cap"),
                "total_volume": coin.get("total_volume"),
                "price_change_24h": coin.get("price_change_percentage_24h"),
                "price_change_7d": coin.get("price_change_percentage_7d_in_currency"),
            })
        return parsed


async def test_coingecko():
    """Test CoinGecko client"""
    client = CoinGeckoClient()
    try:
        coins = await client.get_top_coins(10)
        parsed = client.parse_top_coins(coins)
        for coin in parsed:
            print(f"{coin['market_cap_rank']}. {coin['symbol']}: ${coin['current_price']}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_coingecko())

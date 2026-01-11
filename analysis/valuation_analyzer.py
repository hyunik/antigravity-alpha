"""
Valuation Analyzer
Analyzes coin fundamentals: Market Cap, FDV, Token Unlocks
"""

import asyncio
import aiohttp
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class ValuationData:
    """Coin valuation data"""
    symbol: str
    market_cap: float
    fully_diluted_valuation: float
    circulating_supply: float
    total_supply: float
    max_supply: Optional[float]
    
    # Derived metrics
    fdv_mcap_ratio: float  # FDV / MCap - lower is better
    circulating_ratio: float  # Circulating / Total (0-1)
    
    # Scores
    valuation_score: float  # 0-100
    unlock_risk: str  # "LOW", "MEDIUM", "HIGH"
    summary: str


class ValuationAnalyzer:
    """
    Valuation Analyzer for cryptocurrency fundamentals
    
    Key Metrics:
    - Market Cap: Current market value
    - FDV (Fully Diluted Valuation): Value if all tokens circulating
    - Circulating Ratio: % of tokens already in circulation
    - Unlock Risk: Risk from future token releases
    """
    
    COINGECKO_BASE = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_coin_data(self, coin_id: str) -> Optional[Dict]:
        """
        Fetch coin data from CoinGecko
        
        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum")
            
        Returns:
            Dict with coin market data
        """
        session = await self._get_session()
        url = f"{self.COINGECKO_BASE}/coins/{coin_id}"
        
        try:
            await asyncio.sleep(1.5)  # Rate limiting
            async with session.get(url, params={
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false"
            }) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("CoinGecko rate limited")
                    await asyncio.sleep(60)
                    return None
                else:
                    logger.error(f"CoinGecko error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Failed to fetch coin data: {e}")
            return None
    
    def analyze_from_data(self, data: Dict) -> ValuationData:
        """
        Analyze valuation from CoinGecko data
        
        Args:
            data: CoinGecko coin data
            
        Returns:
            ValuationData analysis
        """
        market_data = data.get("market_data", {})
        
        symbol = data.get("symbol", "").upper()
        market_cap = market_data.get("market_cap", {}).get("usd", 0) or 0
        fdv = market_data.get("fully_diluted_valuation", {}).get("usd", 0) or 0
        circulating = market_data.get("circulating_supply", 0) or 0
        total = market_data.get("total_supply", 0) or 0
        max_supply = market_data.get("max_supply")
        
        # Calculate ratios
        fdv_mcap_ratio = fdv / market_cap if market_cap > 0 else 1
        circulating_ratio = circulating / total if total > 0 else 1
        
        # Valuation score (higher circulating ratio = better)
        valuation_score = 50
        
        # Circulating ratio contribution (max 40 points)
        if circulating_ratio >= 0.9:
            valuation_score += 40  # Fully diluted
        elif circulating_ratio >= 0.7:
            valuation_score += 30
        elif circulating_ratio >= 0.5:
            valuation_score += 20
        elif circulating_ratio >= 0.3:
            valuation_score += 10
        else:
            valuation_score -= 10  # High unlock risk
        
        # FDV/MCap ratio (lower is better)
        if fdv_mcap_ratio <= 1.1:
            valuation_score += 10
        elif fdv_mcap_ratio <= 1.5:
            valuation_score += 5
        elif fdv_mcap_ratio >= 3:
            valuation_score -= 10
        
        # Unlock risk assessment
        if circulating_ratio >= 0.9:
            unlock_risk = "LOW"
            risk_desc = "거의 100% 유통으로 언락 리스크 미미"
        elif circulating_ratio >= 0.7:
            unlock_risk = "MEDIUM"
            risk_desc = f"약 {(1-circulating_ratio)*100:.0f}% 언락 예정"
        else:
            unlock_risk = "HIGH"
            risk_desc = f"⚠️ {(1-circulating_ratio)*100:.0f}% 미유통, 향후 대량 언락 주의"
        
        # Generate summary
        summary = f"유통률 {circulating_ratio*100:.1f}% | FDV/MCap {fdv_mcap_ratio:.2f}x | {risk_desc}"
        
        return ValuationData(
            symbol=symbol,
            market_cap=market_cap,
            fully_diluted_valuation=fdv,
            circulating_supply=circulating,
            total_supply=total,
            max_supply=max_supply,
            fdv_mcap_ratio=fdv_mcap_ratio,
            circulating_ratio=circulating_ratio,
            valuation_score=min(max(valuation_score, 0), 100),
            unlock_risk=unlock_risk,
            summary=summary
        )
    
    async def analyze(self, coin_id: str) -> Optional[ValuationData]:
        """
        Full valuation analysis for a coin
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            ValuationData or None
        """
        data = await self.get_coin_data(coin_id)
        if not data:
            return None
        return self.analyze_from_data(data)
    
    def analyze_from_market_data(
        self,
        symbol: str,
        market_cap: float,
        fdv: float,
        circulating_supply: float,
        total_supply: float
    ) -> ValuationData:
        """
        Analyze valuation from provided market data (no API call)
        
        Useful when data is already available from other sources
        """
        data = {
            "symbol": symbol,
            "market_data": {
                "market_cap": {"usd": market_cap},
                "fully_diluted_valuation": {"usd": fdv},
                "circulating_supply": circulating_supply,
                "total_supply": total_supply,
                "max_supply": None
            }
        }
        return self.analyze_from_data(data)


async def test_valuation():
    """Test valuation analyzer"""
    analyzer = ValuationAnalyzer()
    
    try:
        # Test with provided data (no API call)
        result = analyzer.analyze_from_market_data(
            symbol="WIF",
            market_cap=3_000_000_000,
            fdv=3_000_000_000,
            circulating_supply=998_000_000,
            total_supply=1_000_000_000
        )
        
        print(f"Symbol: {result.symbol}")
        print(f"Market Cap: ${result.market_cap:,.0f}")
        print(f"FDV: ${result.fully_diluted_valuation:,.0f}")
        print(f"Circulating Ratio: {result.circulating_ratio*100:.1f}%")
        print(f"FDV/MCap Ratio: {result.fdv_mcap_ratio:.2f}x")
        print(f"Valuation Score: {result.valuation_score}")
        print(f"Unlock Risk: {result.unlock_risk}")
        print(f"Summary: {result.summary}")
        
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(test_valuation())

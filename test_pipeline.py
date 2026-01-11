#!/usr/bin/env python3
"""
Test script for data pipeline
"""

import asyncio
import sys
sys.path.insert(0, '.')

async def test_pipeline():
    print("=" * 50)
    print("Antigravity-Alpha 데이터 파이프라인 테스트")
    print("=" * 50)
    
    # Test 1: CoinGecko Client
    print("\n[1/3] CoinGecko API 테스트...")
    try:
        from pipeline.coingecko_client import CoinGeckoClient
        client = CoinGeckoClient(rate_limit_delay=1.0)
        
        coins = await client.get_top_coins(5)
        parsed = client.parse_top_coins(coins)
        
        print(f"✅ CoinGecko 연결 성공! 상위 5개 코인:")
        for coin in parsed:
            print(f"   {coin['market_cap_rank']}. {coin['symbol']}: ${coin['current_price']:,.2f}")
        
        await client.close()
    except Exception as e:
        print(f"❌ CoinGecko 오류: {e}")
        return False
    
    # Test 2: Binance Client
    print("\n[2/3] Binance API 테스트...")
    try:
        from pipeline.binance_client import BinanceClient
        client = BinanceClient(rate_limit_delay=0.1)
        
        # Get available pairs
        pairs = await client.get_usdt_pairs()
        print(f"✅ Binance Futures 페어 수: {len(pairs)}개")
        
        # Get OHLCV data
        df = await client.get_ohlcv_df("BTCUSDT", "4h", 20)
        print(f"✅ BTCUSDT 4h 데이터: {len(df)}개 캔들")
        print(f"   최신 캔들: {df.iloc[-1]['timestamp']} | Close: ${df.iloc[-1]['close']:,.2f}")
        
        # Get market data
        market = await client.get_market_data("BTCUSDT")
        print(f"✅ 시장 데이터:")
        print(f"   Open Interest: {market['open_interest']:,.0f}")
        print(f"   Funding Rate: {market['funding_rate']:.6f}")
        
        await client.close()
    except Exception as e:
        print(f"❌ Binance 오류: {e}")
        return False
    
    # Test 3: Data Collector
    print("\n[3/3] 통합 DataCollector 테스트...")
    try:
        from pipeline.data_collector import DataCollector
        collector = DataCollector()
        
        await collector.initialize()
        print(f"✅ DataCollector 초기화 완료")
        
        # Get top coins with Binance mapping
        coins = await collector.get_top_coins(10)
        print(f"✅ 거래 가능 코인: {len(coins)}개")
        
        # Test multi-timeframe data
        if coins:
            symbol = coins[0]["binance_symbol"]
            data = await collector.fetch_multi_timeframe_data(symbol, ["4h"])
            print(f"✅ {symbol} 멀티 타임프레임 데이터: {list(data.keys())}")
        
        await collector.close()
    except Exception as e:
        print(f"❌ DataCollector 오류: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 모든 테스트 통과! 데이터 파이프라인 정상 작동")
    print("=" * 50)
    return True

if __name__ == "__main__":
    result = asyncio.run(test_pipeline())
    sys.exit(0 if result else 1)

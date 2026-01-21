#!/usr/bin/env python3
"""
Quick Test Script for Binance Integration
------------------------------------------
Runs basic tests of the Binance regime trader components.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


RUN_LIVE = os.getenv("RUN_LIVE_BINANCE_TESTS") == "1"


@pytest.mark.asyncio
async def test_binance_client():
    """Test BinanceClient initialization and methods."""
    print("\n" + "=" * 80)
    print("TEST 1: BinanceClient")
    print("=" * 80)
    
    from data.binance_client import BinanceClient
    
    client = BinanceClient(
        api_key='test_key',
        secret_key='test_secret',
        testnet=True,
        environment='paper'
    )
    
    print(f"✓ Client created: {client}")
    
    await client.initialize()
    print("✓ Client initialized in paper mode")
    
    # Test ticker fetch
    ticker = await client.fetch_ticker('BTC/USDT')
    print(f"✓ BTC/USDT ticker: ${ticker['last']:,.2f}")
    
    # Test multi-ticker fetch
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    tickers = await client.fetch_multiple_tickers(symbols)
    print(f"✓ Fetched {len(tickers)} tickers")
    
    # Test OHLCV fetch
    df = await client.fetch_ohlcv('BTC/USDT', '1h', 100)
    print(f"✓ Fetched {len(df)} OHLCV candles")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Test order simulation
    order = await client.place_order('BTC/USDT', 'buy', 'market', 0.001)
    print(f"✓ Simulated order: {order['id']}")
    
    await client.close()
    print("✓ Client closed")
    
    return True


@pytest.mark.skipif(not RUN_LIVE, reason="Live Binance integration disabled; set RUN_LIVE_BINANCE_TESTS=1 to run")
@pytest.mark.asyncio
async def test_regime_trader():
    """Test BinanceRegimeTrader."""
    print("\n" + "=" * 80)
    print("TEST 2: BinanceRegimeTrader")
    print("=" * 80)
    
    from binance_regime_trader import BinanceRegimeTrader
    
    trader = BinanceRegimeTrader(config_path='core/config.yaml')
    print(f"✓ Trader created")
    print(f"✓ Config loaded: {len(trader.config)} sections")
    
    symbols = trader.config.get('trading', {}).get('symbols', [])
    print(f"✓ Configured symbols: {symbols}")
    
    await trader.initialize()
    print(f"✓ Initialized {len(trader.strategies)} strategies")
    
    # Test price fetching
    prices = await trader.fetch_current_prices()
    print(f"✓ Current prices:")
    for symbol, price in prices.items():
        print(f"    {symbol}: ${price:,.2f}")
    
    # Test historical data fetch
    ohlcv = await trader.fetch_historical_data('BTC/USDT', '1h', 100)
    print(f"✓ Fetched {len(ohlcv)} historical candles")
    
    await trader.close()
    print("✓ Trader closed")
    
    return True


@pytest.mark.skipif(not RUN_LIVE, reason="Live Binance integration disabled; set RUN_LIVE_BINANCE_TESTS=1 to run")
@pytest.mark.asyncio
async def test_full_backtest():
    """Test full multi-pair backtest."""
    print("\n" + "=" * 80)
    print("TEST 3: Full Multi-Pair Backtest")
    print("=" * 80)
    
    from binance_regime_trader import main
    
    # Run main function
    await main()
    
    # Check output file
    summary_file = Path('logs/backtest_summary.txt')
    if summary_file.exists():
        print(f"\n✓ Summary file created: {summary_file}")
        print(f"  Size: {summary_file.stat().st_size} bytes")
    else:
        print(f"\n✗ Summary file not found!")
        return False
    
    return True


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BINANCE INTEGRATION TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("BinanceClient", test_binance_client),
        ("BinanceRegimeTrader", test_regime_trader),
        ("Full Backtest", test_full_backtest),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

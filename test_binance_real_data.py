#!/usr/bin/env python3
"""
Quick test to verify real Binance data fetching works
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from data.exchange import ExchangeConnector

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    """Test real Binance data fetch"""
    
    # Create connector to real Binance
    connector = ExchangeConnector(
        exchange_id='binance',
        api_key=None,
        secret=None,
        mode='paper'  # Mode is for execution, not data
    )
    
    print("Connecting to Binance...")
    await connector.connect()
    print("✅ Connected to Binance")
    
    # Calculate since time (500 hours = ~21 days ago)
    now = datetime.now(timezone.utc)
    since_time = now - timedelta(hours=500)
    since_ms = int(since_time.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)
    
    print(f"\nFetching BTC/USDT 1h data:")
    print(f"  Since: {since_time.isoformat()}")
    print(f"  Until: {now.isoformat()}")
    print(f"  Requesting: 500 candles")
    
    try:
        candles = await connector.fetch_ohlcv_range(
            symbol='BTC/USDT',
            timeframe='1h',
            since_ms=since_ms,
            until_ms=until_ms,
            page_limit=500
        )
        
        print(f"\n✅ Received {len(candles)} candles")
        if candles:
            from datetime import datetime as dt_class
            first = candles[0]
            last = candles[-1]
            first_dt = dt_class.fromtimestamp(first['timestamp']/1000, tz=timezone.utc)
            last_dt = dt_class.fromtimestamp(last['timestamp']/1000, tz=timezone.utc)
            print(f"   First: {first_dt.isoformat()} @ ${first['close']:.2f}")
            print(f"   Last:  {last_dt.isoformat()} @ ${last['close']:.2f}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Failed to fetch data")
    finally:
        print("\nDisconnecting...")
        await connector.disconnect()
        print("✅ Disconnected")

if __name__ == '__main__':
    asyncio.run(main())

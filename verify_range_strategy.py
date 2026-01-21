#!/usr/bin/env python3
"""Verify RangeTradingStrategy loads correctly"""

import sys
import asyncio

async def main():
    try:
        from main import TradingPlatform
        
        print("\n" + "="*70)
        print("TESTING RANGE TRADING STRATEGY INTEGRATION")
        print("="*70)
        
        # Create platform with test config
        platform = TradingPlatform(
            config_path="FIXED_CONFIG_FOR_VOLATILE_MARKETS.yaml",
            run_once=True
        )
        
        print("\n[1] Loading configuration...")
        platform.config = platform.load_config()
        print(f"✅ Config loaded")
        
        print("\n[2] Initializing core modules...")
        await platform._setup_core()
        print(f"✅ Core initialized")
        
        print("\n[3] Initializing monitoring...")
        await platform._setup_monitoring()
        print(f"✅ Monitoring initialized")
        
        print("\n[4] Initializing data modules...")
        await platform._setup_data()
        print(f"✅ Data modules initialized")
        
        print("\n[5] Initializing strategy...")
        await platform._setup_strategy()
        print(f"✅ Strategy initialized")
        print(f"   Strategy type: {type(platform.strategy).__name__}")
        print(f"   Strategy name: {platform.strategy.name}")
        
        # Check if it's a RegimeBasedStrategy with MultiStrategyAggregator
        if hasattr(platform.strategy, 'underlying_strategy'):
            underlying = platform.strategy.underlying_strategy
            print(f"\n   Underlying strategy: {type(underlying).__name__}")
            
            if hasattr(underlying, 'strategies'):
                print(f"   Aggregated strategies ({len(underlying.strategies)}):")
                for s in underlying.strategies:
                    weight = underlying.weights.get(s.name, 1.0)
                    print(f"      - {s.name} (weight: {weight})")
        
        print("\n" + "="*70)
        print("✅ STRATEGY INTEGRATION SUCCESSFUL")
        print("="*70)
        
        # Show loaded config for strategies
        print("\nMulti-strategy configuration:")
        multi_cfg = platform.config.get('strategy', {}).get('multi_strategy', {})
        if multi_cfg.get('enabled'):
            for strat_cfg in multi_cfg.get('strategies', []):
                name = strat_cfg.get('name')
                enabled = strat_cfg.get('enabled', True)
                weight = strat_cfg.get('weight', 1.0)
                status = "✅" if enabled else "❌"
                print(f"  {status} {name:20s} weight={weight:.2f}")
        
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

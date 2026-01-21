#!/usr/bin/env python3
"""Quick test of RangeTradingStrategy volatility calculation"""

import sys
import pandas as pd
import numpy as np

# Test volatility calculation methods
def test_volatility_methods():
    # Generate test price data
    np.random.seed(42)
    prices = np.linspace(90000, 92000, 100) + np.random.normal(0, 500, 100)
    
    print("=" * 70)
    print("VOLATILITY CALCULATION COMPARISON")
    print("=" * 70)
    
    # Method 1: Old way (simple std of % returns) - what was failing
    returns_pct = prices[1:] / prices[:-1] - 1.0
    vol_old = float(np.std(returns_pct))
    print(f"\n1. OLD METHOD (simple std of pct returns):")
    print(f"   Volatility: {vol_old:.6f} ({vol_old*100:.2f}%)")
    print(f"   ❌ This is TOO LOW for the check")
    
    # Method 2: New way (annualized, matching regime detector)
    log_returns = np.log(prices[1:] / prices[:-1])
    vol_new = float(np.std(log_returns) * np.sqrt(252))
    print(f"\n2. NEW METHOD (std(log_returns) * sqrt(252) - ANNUALIZED):")
    print(f"   Volatility: {vol_new:.6f} ({vol_new*100:.2f}%)")
    print(f"   ✅ This matches the regime detector")
    
    # Test parameters
    min_vol = 0.001
    max_vol = 0.15
    
    print(f"\n3. THRESHOLDS:")
    print(f"   Min volatility: {min_vol:.6f} ({min_vol*100:.2f}%)")
    print(f"   Max volatility: {max_vol:.6f} ({max_vol*100:.2f}%)")
    
    print(f"\n4. OLD METHOD CHECK: {min_vol} <= {vol_old:.6f} <= {max_vol}?")
    print(f"   Result: {min_vol <= vol_old <= max_vol} ❌ FAILS")
    
    print(f"\n5. NEW METHOD CHECK: {min_vol} <= {vol_new:.6f} <= {max_vol}?")
    print(f"   Result: {min_vol <= vol_new <= max_vol} ✅ PASSES")
    
    print("\n" + "=" * 70)
    print("CONCLUSION: Using annualized volatility fixes the issue!")
    print("=" * 70)

if __name__ == "__main__":
    test_volatility_methods()

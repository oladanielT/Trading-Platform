#!/usr/bin/env python3
"""
Quick analysis of paper trading session logs
"""
import re
import sys
from pathlib import Path


def analyze_logs(log_file="logs/paper_first_test.log"):
    """Parse logs and extract key metrics"""
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}")
        print("   Make sure you've run a paper trading session first")
        return False
    
    with open(log_file, 'r') as f:
        logs = f.read()
    
    # Extract metrics using patterns
    signals = re.findall(r'Signal: (\w+)', logs, re.IGNORECASE)
    buy_signals = [s for s in signals if s.upper() == 'BUY']
    sell_signals = [s for s in signals if s.upper() == 'SELL']
    
    # Position sizing
    sizing_approved = re.findall(r'approved.*?True', logs, re.IGNORECASE)
    sizing_rejected = re.findall(r'approved.*?False', logs, re.IGNORECASE)
    
    # Orders
    orders_placed = re.findall(r'Submitting.*order', logs, re.IGNORECASE)
    orders_filled = re.findall(r'Order filled|filled.*order', logs, re.IGNORECASE)
    
    # Positions
    positions_opened = re.findall(r'Position opened|Opening position', logs, re.IGNORECASE)
    positions_closed = re.findall(r'Position closed|Closing position', logs, re.IGNORECASE)
    
    # Try to extract PnL (multiple patterns)
    pnl_patterns = [
        r'pnl[:\s]+([-\d.]+)',
        r'profit[:\s]+([-\d.]+)',
        r'P&L[:\s]+([-\d.]+)',
    ]
    pnls = []
    for pattern in pnl_patterns:
        pnls.extend(re.findall(pattern, logs, re.IGNORECASE))
    
    pnls = [float(p) for p in pnls if p]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    # Errors
    errors = re.findall(r'ERROR[:\s]+(.*)', logs)
    warnings = re.findall(r'WARNING[:\s]+(.*)', logs)
    
    # Calculate metrics
    total_signals = len(signals)
    total_sizing = len(sizing_approved) + len(sizing_rejected)
    total_orders = len(orders_placed)
    total_fills = len(orders_filled)
    total_positions_opened = len(positions_opened)
    total_positions_closed = len(positions_closed)
    total_trades = len(pnls)
    
    sizing_approval_rate = len(sizing_approved) / total_sizing if total_sizing > 0 else 0
    fill_rate = total_fills / total_orders if total_orders > 0 else 0
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    total_pnl = sum(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    
    # Try to find final equity
    equity_matches = re.findall(r'equity[:\s]+([\d.]+)', logs, re.IGNORECASE)
    final_equity = float(equity_matches[-1]) if equity_matches else 1000.0
    
    # Report
    print("=" * 70)
    print("PAPER TRADING SESSION ANALYSIS")
    print("=" * 70)
    print(f"Log file: {log_file}")
    print(f"Log size: {log_path.stat().st_size / 1024:.1f} KB")
    print("=" * 70)
    
    print("\nSIGNAL GENERATION:")
    print(f"  Total signals:        {total_signals}")
    print(f"    ├─ BUY signals:     {len(buy_signals)}")
    print(f"    └─ SELL signals:    {len(sell_signals)}")
    
    print("\nRISK MANAGEMENT:")
    print(f"  Position sizing runs: {total_sizing}")
    print(f"    ├─ Approved:        {len(sizing_approved)} ({sizing_approval_rate:.0%})")
    print(f"    └─ Rejected:        {len(sizing_rejected)}")
    
    print("\nORDER EXECUTION:")
    print(f"  Orders placed:        {total_orders}")
    print(f"  Orders filled:        {total_fills}")
    print(f"  Fill rate:            {fill_rate:.0%}")
    
    print("\nPOSITION TRACKING:")
    print(f"  Positions opened:     {total_positions_opened}")
    print(f"  Positions closed:     {total_positions_closed}")
    
    if total_trades > 0:
        print("\nTRADE PERFORMANCE:")
        print(f"  Total trades:         {total_trades}")
        print(f"    ├─ Wins:            {len(wins)}")
        print(f"    └─ Losses:          {len(losses)}")
        print(f"  Win rate:             {win_rate:.1%}")
        print("-" * 70)
        print(f"  Total PnL:            ${total_pnl:.2f}")
        print(f"  Average win:          ${avg_win:.2f}")
        print(f"  Average loss:         ${avg_loss:.2f}")
        print(f"  Profit factor:        {profit_factor:.2f}")
    else:
        print("\nTRADE PERFORMANCE:")
        print("  No trades closed yet")
    
    print("\nEQUITY:")
    print(f"  Starting equity:      $1000.00")
    print(f"  Final equity:         ${final_equity:.2f}")
    print(f"  Return:               {((final_equity - 1000) / 1000) * 100:.2f}%")
    
    # Errors and warnings
    print("\nSYSTEM HEALTH:")
    if errors:
        print(f"  ⚠️  {len(errors)} errors detected")
        for i, error in enumerate(errors[:3], 1):
            print(f"    {i}. {error[:70]}")
        if len(errors) > 3:
            print(f"    ... and {len(errors) - 3} more")
    else:
        print("  ✓ No errors detected")
    
    if warnings:
        print(f"  ⚠️  {len(warnings)} warnings")
    
    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK:")
    print("=" * 70)
    
    checks = [
        ("At least 1 signal generated", total_signals >= 1),
        ("At least 1 position sizing approved", len(sizing_approved) >= 1),
        ("At least 1 order placed", total_orders >= 1),
        ("At least 1 order filled", total_fills >= 1),
        ("Equity changed from starting value", final_equity != 1000.0),
        ("No critical errors", len(errors) == 0),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("✓ EXECUTION FLOW VERIFIED - Ready for longer paper sessions")
        print("\nNext steps:")
        print("  1. Run 24-hour session with this config")
        print("  2. Enable multi-strategy aggregator")
        print("  3. Build TradeJournal for automated metrics")
    else:
        print("❌ EXECUTION ISSUES DETECTED - Debug before proceeding")
        print("\nTroubleshooting:")
        if total_signals == 0:
            print("  • No signals: Check if strategy logic is working")
        if len(sizing_approved) == 0 and total_signals > 0:
            print("  • Sizing rejections: Check stop_loss metadata and risk settings")
        if total_orders == 0 and len(sizing_approved) > 0:
            print("  • No orders: Check broker connection/execution layer")
        if total_fills == 0 and total_orders > 0:
            print("  • No fills: Check paper broker is actually filling orders")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = analyze_logs()
    sys.exit(0 if success else 1)

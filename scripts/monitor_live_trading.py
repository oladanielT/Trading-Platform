#!/usr/bin/env python3
"""
monitor_live_trading.py - Integrated live trading monitoring dashboard

Combines all monitoring scripts into a single command-line interface:
- Trade success metrics
- Drawdown analysis
- Readiness correlation
- Real-time alerts

Run this periodically (e.g., hourly) to check trading performance.
"""

import sys
import argparse
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from monitor_trade_success import TradeSuccessMonitor
from monitor_drawdown import DrawdownMonitor
from monitor_readiness_correlation import ReadinessCorrelationAnalyzer


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Main monitoring dashboard."""
    parser = argparse.ArgumentParser(
        description='Live trading monitoring dashboard'
    )
    parser.add_argument(
        '--log',
        type=str,
        default='logs/live_trading.log',
        help='Path to trading log file'
    )
    parser.add_argument(
        '--section',
        type=str,
        choices=['trades', 'drawdown', 'readiness', 'all'],
        default='all',
        help='Which section to display'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--initial-equity',
        type=float,
        default=100000.0,
        help='Starting account equity'
    )
    
    args = parser.parse_args()
    
    # Validate log file exists
    if not Path(args.log).exists():
        print(f"Error: Log file not found: {args.log}")
        print("\nMake sure you have:")
        print("1. Started the trading bot with paper/testnet/live mode")
        print("2. Specified the correct log file path")
        sys.exit(1)
    
    if args.json:
        # JSON output mode
        import json
        output = {}
        
        if args.section in ['trades', 'all']:
            monitor = TradeSuccessMonitor(log_file=args.log)
            monitor.load_trades_from_log()
            output['trades'] = monitor.calculate_metrics()
        
        if args.section in ['readiness', 'all']:
            analyzer = ReadinessCorrelationAnalyzer()
            analyzer.load_trades_from_log(args.log)
            output['readiness'] = analyzer.analyze_by_readiness()
            output['insights'] = analyzer.calculate_correlation_insights()
        
        if args.section in ['drawdown', 'all']:
            monitor = DrawdownMonitor(initial_equity=args.initial_equity)
            output['drawdown'] = monitor.calculate_drawdown_metrics()
            output['limits'] = monitor.check_drawdown_limits()
        
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output mode
        print_section("LIVE TRADING MONITORING DASHBOARD")
        print(f"Log file: {args.log}")
        print(f"Generated: {__import__('datetime').datetime.utcnow().isoformat()}\n")
        
        if args.section in ['trades', 'all']:
            print_section("TRADE SUCCESS ANALYSIS")
            monitor = TradeSuccessMonitor(log_file=args.log)
            if monitor.load_trades_from_log():
                print(monitor.generate_report())
            else:
                print("No trades recorded yet.\n")
        
        if args.section in ['drawdown', 'all']:
            print_section("DRAWDOWN ANALYSIS")
            monitor = DrawdownMonitor(initial_equity=args.initial_equity)
            print(monitor.generate_report(
                max_daily_dd=0.05,
                max_total_dd=0.20
            ))
        
        if args.section in ['readiness', 'all']:
            print_section("READINESS STATE CORRELATION")
            analyzer = ReadinessCorrelationAnalyzer()
            if analyzer.load_trades_from_log(args.log):
                print(analyzer.generate_report())
            else:
                print("No trades recorded yet.\n")
        
        print_section("QUICK REFERENCE")
        print("""
PAPER MODE VALIDATION CHECKLIST
================================
□ Run for 1-2 weeks minimum
□ Check win rate >= 50% (target 55%+)
□ Verify max drawdown < 2% on good days
□ Confirm readiness state matches expected patterns
□ Monitor trade distribution across symbols
□ Validate alert system (check logs for [Alert] markers)

LIVE ACTIVATION CHECKLIST
==========================
□ Paper results stable and positive
□ Risk parameters configured: max 10% position, 2% risk
□ API credentials verified (testnet/live)
□ Position limits and drawdown guards active
□ Logging configured for all trades
□ Monitoring scripts tested
□ Read LIVE_TRADING_GUIDE.md thoroughly

TROUBLESHOOTING
===============
Win rate too low (< 45%)?
  → Check regime_confidence_threshold
  → Review allowed_regimes configuration
  → Verify underlying strategy parameters

Too many alerts?
  → Increase confidence_threshold in alerts config
  → Increase deviation_threshold
  → Check readiness_alert_threshold

Unexpected drawdowns?
  → Review risk per trade (default 2%)
  → Check max_position_size limit (default 10%)
  → Verify position sizing method
  → Check slippage assumptions in backtests
""")


if __name__ == '__main__':
    main()

#!/bin/bash
# LIVE_TRADING_SCRIPTS.sh - Quick reference for all monitoring scripts
#
# This file demonstrates how to use all four monitoring scripts
# Copy these commands to quickly check trading status

# Make this file executable:
# chmod +x LIVE_TRADING_SCRIPTS.sh

echo "Live Trading Monitoring Scripts"
echo "==============================="
echo ""

# 1. FULL MONITORING DASHBOARD
echo "1. Full Monitoring Dashboard (all metrics)"
echo "   Command: python scripts/monitor_live_trading.py --section all"
echo "   Shows:   Trades, drawdown, readiness, alerts"
echo "   Time:    ~2 seconds"
echo ""

# 2. TRADE SUCCESS MONITORING  
echo "2. Trade Success Metrics"
echo "   Command: python scripts/monitor_live_trading.py --section trades"
echo "   Shows:   Win rate, profit factor, consecutive wins/losses"
echo "   Time:    <1 second"
echo ""

# 3. DRAWDOWN MONITORING
echo "3. Drawdown Analysis"
echo "   Command: python scripts/monitor_live_trading.py --section drawdown"
echo "   Shows:   Max DD, daily DD, recovery status"
echo "   Time:    <1 second"
echo ""

# 4. READINESS CORRELATION
echo "4. Readiness State Correlation"
echo "   Command: python scripts/monitor_live_trading.py --section readiness"
echo "   Shows:   Win rate by readiness, state transitions"
echo "   Time:    <1 second"
echo ""

# 5. JSON OUTPUT
echo "5. JSON Output (for analysis)"
echo "   Command: python scripts/monitor_live_trading.py --json > metrics.json"
echo "   Shows:   All metrics in JSON format"
echo "   Use for: Automated analysis, graphing"
echo ""

# QUICK HEALTH CHECKS
echo "======================================"
echo "QUICK HEALTH CHECKS"
echo "======================================"
echo ""

echo "30-second health check:"
echo "  python scripts/monitor_live_trading.py --section trades"
echo "  python scripts/monitor_live_trading.py --section drawdown"
echo ""

echo "Watch logs in real-time:"
echo "  tail -f logs/live_trading.log"
echo ""

echo "Filter logs for specific events:"
echo "  tail -f logs/live_trading.log | grep 'Trade closed'   # Show trades"
echo "  tail -f logs/live_trading.log | grep '\[Alert\]'      # Show alerts"
echo "  tail -f logs/live_trading.log | grep 'Readiness'      # Show state changes"
echo "  tail -f logs/live_trading.log | grep 'ERROR'          # Show errors"
echo ""

# DAILY WORKFLOW
echo "======================================"
echo "DAILY WORKFLOW"
echo "======================================"
echo ""

echo "Morning (09:00 AM):"
echo "  tail logs/live_trading.log | tail -50"
echo ""

echo "Mid-morning (10:00 AM):"
echo "  python scripts/monitor_live_trading.py --section trades"
echo ""

echo "Afternoon (01:00 PM):"
echo "  python scripts/monitor_live_trading.py --section drawdown"
echo ""

echo "End of day (04:00 PM):"
echo "  python scripts/monitor_live_trading.py --section all"
echo ""

echo "Weekly (Friday):"
echo "  python scripts/monitor_readiness_correlation.py"
echo "  python scripts/monitor_trade_success.py"
echo "  python scripts/monitor_drawdown.py"
echo ""

# AUTOMATED MONITORING
echo "======================================"
echo "AUTOMATED MONITORING (Optional)"
echo "======================================"
echo ""

echo "Setup hourly monitoring with cron:"
echo "  crontab -e"
echo "  # Add this line:"
echo "  0 * * * * python /path/to/scripts/monitor_live_trading.py --section all >> /tmp/trading_monitor.log 2>&1"
echo ""

echo "Watch automated monitoring:"
echo "  tail -f /tmp/trading_monitor.log"
echo ""

# EXAMPLES
echo "======================================"
echo "EXAMPLE USAGE"
echo "======================================"
echo ""

echo "Example 1: Check if ready to scale capital"
echo "  $ python scripts/monitor_live_trading.py --section trades"
echo ""
echo "  Output shows:"
echo "    Win rate:       56% (✓ > 50%)"
echo "    Net PnL:        +$2,847"
echo "    Profit factor:  1.45"
echo "  Decision: ✓ Ready to increase capital"
echo ""

echo "Example 2: Emergency drawdown check"
echo "  $ python scripts/monitor_live_trading.py --section drawdown"
echo ""
echo "  Output shows:"
echo "    Max drawdown:   8.2% (✗ > 5% limit)"
echo "  Decision: ✗ HALT trading, reduce capital"
echo ""

echo "Example 3: Readiness state validation"
echo "  $ python scripts/monitor_live_trading.py --section readiness"
echo ""
echo "  Output shows:"
echo "    Ready wins:     60% (3 trades)"
echo "    Forming wins:   45% (9 trades)"
echo "    Not_Ready wins: 20% (5 trades)"
echo "  Decision: ✓ Readiness gating working properly"
echo ""

# HELP
echo "======================================"
echo "HELP & TROUBLESHOOTING"
echo "======================================"
echo ""

echo "Get help for any script:"
echo "  python scripts/monitor_live_trading.py --help"
echo "  python scripts/monitor_trade_success.py --help"
echo "  python scripts/monitor_drawdown.py --help"
echo "  python scripts/monitor_readiness_correlation.py --help"
echo ""

echo "Export to JSON for analysis:"
echo "  python scripts/monitor_live_trading.py --json > analysis.json"
echo "  python scripts/monitor_trade_success.py --json > trades.json"
echo "  python scripts/monitor_readiness_correlation.py --json > readiness.json"
echo ""

echo "No trades yet?"
echo "  Make sure you have:"
echo "  1. Started the trading bot (python main.py)"
echo "  2. Let it run for at least 5-10 minutes"
echo "  3. Check logs exist: ls -la logs/live_trading.log"
echo ""

echo "======================================"
echo "FILES CREATED/MODIFIED"
echo "======================================"
echo ""
echo "Scripts (in scripts/ directory):"
echo "  ✓ monitor_live_trading.py           - Main dashboard"
echo "  ✓ monitor_trade_success.py          - Trade analysis"
echo "  ✓ monitor_drawdown.py               - Drawdown tracking"
echo "  ✓ monitor_readiness_correlation.py  - Readiness analysis"
echo ""
echo "Configuration:"
echo "  ✓ config-live-trading.yaml          - Live trading template"
echo "  ✓ core/config.yaml                  - Active configuration"
echo ""
echo "Documentation:"
echo "  ✓ LIVE_TRADING_ACTIVATION_GUIDE.md  - Complete guide"
echo "  ✓ LIVE_TRADING_QUICK_REFERENCE.md   - Quick lookup"
echo "  ✓ LIVE_TRADING_ACTIVATION_IMPLEMENTATION.md - Summary"
echo "  ✓ LIVE_TRADING_SCRIPTS.sh           - This file"
echo ""

echo "======================================"
echo "NEXT STEPS"
echo "======================================"
echo ""
echo "1. Copy config template:"
echo "   cp config-live-trading.yaml core/config.yaml"
echo ""
echo "2. Start trading in paper mode:"
echo "   python main.py"
echo ""
echo "3. Monitor in another terminal:"
echo "   python scripts/monitor_live_trading.py --section all"
echo ""
echo "4. Follow the activation guide:"
echo "   Read LIVE_TRADING_ACTIVATION_GUIDE.md"
echo ""
echo "Good luck! 🚀"

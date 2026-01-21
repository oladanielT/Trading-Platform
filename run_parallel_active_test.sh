#!/bin/bash
# Parallel test launcher - runs FVG strategy while EMA test continues
# Purpose: FVG generates more signals to validate execution flow

CONFIG="configs/config_paper_active_test.yaml"
LOG="logs/paper_active_test.log"

echo "=========================================="
echo "Starting PARALLEL active strategy test (FVG)"
echo "Config: $CONFIG"
echo "Log: $LOG"
echo "=========================================="
echo ""
echo "Running FVG strategy for 6 hours"
echo "FVG should generate 10-50 signals"
echo "This will validate execution flow faster than EMA"
echo ""
echo "PARALLEL TESTS:"
echo "  1. EMA test (original): Waiting for crossovers (low frequency)"
echo "  2. FVG test (this): Pattern-based signals (high frequency)"
echo ""

# Create logs directory
mkdir -p logs

# Clean old logs
rm -f $LOG

# Start FVG bot in background
echo "Launching FVG bot..."
python3 main.py --config $CONFIG > $LOG 2>&1 &
BOT_PID=$!

echo "Bot started (PID: $BOT_PID)"
echo "Check EMA bot: ps aux | grep main.py"
echo ""
echo "Monitoring for 5 minutes..."

# Wait 5 minutes then show stats
sleep 300

echo ""
echo "=========================================="
echo "5-minute checkpoint (FVG strategy)"
echo "=========================================="

echo "Signals: $(grep -c 'Signal:' $LOG 2>/dev/null || echo '0')"
echo "Orders placed: $(grep -c 'Submitting' $LOG 2>/dev/null || echo '0')"
echo "Orders filled: $(grep -c 'Order filled' $LOG 2>/dev/null || echo '0')"
echo "Errors: $(grep -c 'ERROR' $LOG 2>/dev/null || echo '0')"

echo ""
echo "Bot still running. Continue monitoring:"
echo "  tail -f $LOG"
echo ""
echo "To stop: kill $BOT_PID"
echo "To analyze: python3 analyze_paper_session.py (specify log file)"

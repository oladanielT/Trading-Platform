#!/bin/bash
# Quick paper test launcher with real-time monitoring

CONFIG="configs/config_paper_first_test.yaml"
LOG="logs/paper_first_test.log"

echo "=========================================="
echo "Starting 6-hour paper trading test"
echo "Config: $CONFIG"
echo "Log: $LOG"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Clean old logs
rm -f $LOG

# Start trading bot in background
echo "Launching trading bot..."
python3 main.py --config $CONFIG > $LOG 2>&1 &
BOT_PID=$!

echo "Bot started (PID: $BOT_PID)"
echo ""
echo "Monitoring first 30 minutes..."
echo "Press Ctrl+C to stop monitoring (bot will continue running)"
echo ""

# Wait a moment for bot to initialize
sleep 3

# Show real-time log tail
tail -f $LOG &
TAIL_PID=$!

# Set up trap to clean up tail on Ctrl+C
trap "kill $TAIL_PID 2>/dev/null; exit 0" INT TERM

# Wait 30 minutes
sleep 1800

# Stop monitoring
kill $TAIL_PID 2>/dev/null

echo ""
echo "=========================================="
echo "30-minute checkpoint"
echo "=========================================="

# Quick stats
echo "Signals generated: $(grep -c 'Signal:' $LOG)"
echo "Position sizing: $(grep -c 'Position size:' $LOG)"
echo "Orders placed: $(grep -c 'Submitting' $LOG)"
echo "Errors: $(grep -c 'ERROR' $LOG)"

echo ""
echo "Bot still running (PID: $BOT_PID)"
echo "To stop bot: kill $BOT_PID"
echo "To continue monitoring: tail -f $LOG"
echo "To analyze after session: python3 analyze_paper_session.py"

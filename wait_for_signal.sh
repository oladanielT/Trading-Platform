#!/bin/bash
echo "Waiting for first BUY/SELL signal..."
echo "Config: 5m timeframe, EMA 8/21, 1.5× ATR"
echo ""

count=0
while [ true ]; do
    count=$((count + 1))
    
    # Check for signal
    signals=$(grep -E 'Signal: (BUY|SELL)' logs/paper_first_test.log 2>/dev/null | wc -l)
    
    if [ $signals -gt 0 ]; then
        echo ""
        echo "✅ FIRST SIGNAL DETECTED! (After $count checks)"
        echo ""
        grep -E 'Signal: (BUY|SELL)' logs/paper_first_test.log | head -1
        echo ""
        echo "Next: Run ./quick_stats.sh to see positions/fills"
        break
    fi
    
    # Check every 10 seconds, show progress
    if [ $((count % 6)) -eq 0 ]; then
        iterations=$(grep 'Trading Loop Iteration' logs/paper_first_test.log 2>/dev/null | wc -l)
        echo "[$count checks] $iterations iterations, still searching for crossover..."
    fi
    
    sleep 10
done

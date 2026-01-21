#!/bin/bash
LOG="logs/paper_first_test.log"
echo "========================================"
echo "Quick Stats"
echo "========================================"
echo "BUY/SELL Signals:  $(grep -E 'Signal: (BUY|SELL)' $LOG 2>/dev/null | wc -l)"
echo "HOLD Signals:      $(grep 'Signal: HOLD' $LOG 2>/dev/null | wc -l)"
echo "Position Opened:   $(grep 'Position opened' $LOG 2>/dev/null | wc -l)"
echo "Orders Filled:     $(grep 'Order filled' $LOG 2>/dev/null | wc -l)"
echo "Errors:            $(grep 'ERROR' $LOG 2>/dev/null | wc -l)"
echo "Iterations:        $(grep 'Trading Loop Iteration' $LOG 2>/dev/null | wc -l)"
echo "========================================"
if [ $(grep -E 'Signal: (BUY|SELL)' $LOG 2>/dev/null | wc -l) -gt 0 ]; then
    echo "✅ First signal detected!"
    grep -E 'Signal: (BUY|SELL)' $LOG 2>/dev/null | head -1
fi

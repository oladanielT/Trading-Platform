#!/bin/bash
echo "Monitoring for BUY/SELL signals..."
echo "Press Ctrl+C to stop"
echo ""
echo "Real-time signal monitor:"
tail -f logs/paper_first_test.log | grep -E "(BUY|SELL|Signal:|Position opened|filled)"

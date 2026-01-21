#!/usr/bin/env python3
"""
monitor_trade_success.py - Trade success analysis for live trading

Analyzes trades to monitor:
- Win rate and trade count
- Average win/loss sizes
- Profit factor (gross profit / gross loss)
- Consecutive win/loss streaks
- Per-symbol performance
- Confidence correlation with outcomes

Designed for ongoing monitoring during paper/live trading phases.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from statistics import mean, stdev
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeSuccessMonitor:
    """Monitor trade success metrics from live/paper trading logs."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize monitor.
        
        Args:
            log_file: Path to trading log file (optional)
        """
        self.log_file = log_file or "logs/live_trading.log"
        self.trades: List[Dict] = []
        self.alerts: List[str] = []
        
    def load_trades_from_log(self) -> bool:
        """
        Load completed trades from log file.
        
        Returns:
            True if successful, False otherwise
        """
        if not Path(self.log_file).exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return False
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if 'Trade closed' in line or 'trade_closed' in line:
                        try:
                            # Try to parse JSON if present
                            if '{' in line and '}' in line:
                                json_str = line[line.find('{'):line.rfind('}')+1]
                                trade = json.loads(json_str)
                                self.trades.append(trade)
                        except (json.JSONDecodeError, ValueError):
                            pass  # Skip unparseable lines
            
            logger.info(f"Loaded {len(self.trades)} trades from {self.log_file}")
            return len(self.trades) > 0
            
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return False
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate success metrics.
        
        Returns:
            Dictionary with metrics
        """
        if not self.trades:
            return {
                'status': 'no_trades',
                'message': 'No closed trades recorded yet'
            }
        
        # Basic counts
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in self.trades if t.get('pnl', 0) < 0)
        breakeven = sum(1 for t in self.trades if t.get('pnl', 0) == 0)
        
        # PnL analysis
        winning_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [abs(t.get('pnl', 0)) for t in self.trades if t.get('pnl', 0) < 0]
        all_pnl = [t.get('pnl', 0) for t in self.trades]
        
        total_profit = sum(winning_trades) if winning_trades else 0.0
        total_loss = sum(losing_trades) if losing_trades else 0.0
        net_pnl = sum(all_pnl)
        
        # Calculate metrics
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else (1.0 if total_profit > 0 else 0.0)
        avg_win = mean(winning_trades) if winning_trades else 0.0
        avg_loss = mean(losing_trades) if losing_trades else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Streaks
        win_streak, loss_streak = self._calculate_streaks()
        
        # Per-symbol analysis
        symbol_stats = self._analyze_by_symbol()
        
        # Recent performance (last 10 trades)
        recent_trades = self.trades[-10:] if len(self.trades) > 10 else self.trades
        recent_wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        recent_win_rate = recent_wins / len(recent_trades) if recent_trades else 0.0
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'breakeven': breakeven,
            'win_rate': f"{win_rate*100:.1f}%",
            'profit_factor': f"{profit_factor:.2f}",
            'total_profit': f"${total_profit:.2f}",
            'total_loss': f"${total_loss:.2f}",
            'net_pnl': f"${net_pnl:.2f}",
            'avg_win': f"${avg_win:.2f}",
            'avg_loss': f"${avg_loss:.2f}",
            'expectancy': f"${expectancy:.2f}",
            'consecutive_wins': win_streak,
            'consecutive_losses': loss_streak,
            'recent_10_trades_wr': f"{recent_win_rate*100:.1f}%",
            'by_symbol': symbol_stats,
        }
    
    def _calculate_streaks(self) -> Tuple[int, int]:
        """Calculate current win and loss streaks."""
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in self.trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_win_streak = 0
                current_loss_streak = 0
        
        return max_win_streak, max_loss_streak
    
    def _analyze_by_symbol(self) -> Dict[str, Dict]:
        """Analyze performance by trading symbol."""
        by_symbol = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
        
        for trade in self.trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            pnl = trade.get('pnl', 0)
            
            if pnl > 0:
                by_symbol[symbol]['wins'] += 1
            elif pnl < 0:
                by_symbol[symbol]['losses'] += 1
            
            by_symbol[symbol]['pnl'] += pnl
        
        # Format output
        result = {}
        for symbol, stats in sorted(by_symbol.items()):
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total if total > 0 else 0.0
            result[symbol] = {
                'trades': total,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': f"{wr*100:.1f}%",
                'pnl': f"${stats['pnl']:.2f}",
            }
        
        return result
    
    def generate_report(self) -> str:
        """
        Generate human-readable report.
        
        Returns:
            Formatted report string
        """
        metrics = self.calculate_metrics()
        
        if metrics.get('status') == 'no_trades':
            return "No trades to analyze yet. Check back after trading begins."
        
        report = []
        report.append("=" * 80)
        report.append("TRADE SUCCESS ANALYSIS")
        report.append("=" * 80)
        report.append(f"Report generated: {metrics['timestamp']}")
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS")
        report.append("-" * 40)
        report.append(f"Total trades:         {metrics['total_trades']}")
        report.append(f"Wins / Losses:        {metrics['wins']} / {metrics['losses']} (breakeven: {metrics['breakeven']})")
        report.append(f"Win rate:             {metrics['win_rate']}")
        report.append(f"Recent 10 trades WR:  {metrics['recent_10_trades_wr']}")
        report.append("")
        
        # PnL metrics
        report.append("PROFITABILITY")
        report.append("-" * 40)
        report.append(f"Gross profit:         {metrics['total_profit']}")
        report.append(f"Gross loss:           {metrics['total_loss']}")
        report.append(f"Net PnL:              {metrics['net_pnl']}")
        report.append(f"Profit factor:        {metrics['profit_factor']}")
        report.append(f"Avg win:              {metrics['avg_win']}")
        report.append(f"Avg loss:             {metrics['avg_loss']}")
        report.append(f"Expectancy per trade: {metrics['expectancy']}")
        report.append("")
        
        # Streaks
        report.append("STREAKS")
        report.append("-" * 40)
        report.append(f"Max consecutive wins:  {metrics['consecutive_wins']}")
        report.append(f"Max consecutive losses: {metrics['consecutive_losses']}")
        report.append("")
        
        # Per-symbol
        if metrics.get('by_symbol'):
            report.append("PERFORMANCE BY SYMBOL")
            report.append("-" * 40)
            for symbol, stats in metrics['by_symbol'].items():
                report.append(f"{symbol:12} | Trades: {stats['trades']:3} | "
                            f"W/L: {stats['wins']}/{stats['losses']} | "
                            f"WR: {stats['win_rate']:6} | PnL: {stats['pnl']:>10}")
            report.append("")
        
        report.append("=" * 80)
        
        # Check for issues
        if float(metrics['win_rate'].strip('%')) < 45.0:
            self.alerts.append(f"⚠️  Low win rate: {metrics['win_rate']} (target: 55%+)")
        if metrics['consecutive_losses'] >= 5:
            self.alerts.append(f"⚠️  High consecutive losses: {metrics['consecutive_losses']}")
        
        if self.alerts:
            report.append("ALERTS")
            report.append("-" * 40)
            for alert in self.alerts:
                report.append(f"  {alert}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Monitor trade success metrics from live trading logs'
    )
    parser.add_argument(
        '--log',
        type=str,
        default='logs/live_trading.log',
        help='Path to trading log file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    monitor = TradeSuccessMonitor(log_file=args.log)
    
    if not monitor.load_trades_from_log():
        print(f"No trades found in {args.log}")
        sys.exit(1)
    
    if args.json:
        print(json.dumps(monitor.calculate_metrics(), indent=2))
    else:
        print(monitor.generate_report())


if __name__ == '__main__':
    main()

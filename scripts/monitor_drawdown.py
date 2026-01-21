#!/usr/bin/env python3
"""
monitor_drawdown.py - Drawdown analysis and monitoring

Monitors:
- Daily drawdown percentage
- Maximum drawdown from peak (running)
- Drawdown duration
- Recovery time
- Drawdown vs readiness state correlation

Helps validate that observed drawdowns match expected risk parameters.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DrawdownMonitor:
    """Monitor drawdown metrics from equity/trade records."""
    
    def __init__(self, initial_equity: float = 100000.0):
        """
        Initialize monitor.
        
        Args:
            initial_equity: Starting account equity
        """
        self.initial_equity = initial_equity
        self.equity_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[Dict] = []
        
    def record_equity(self, timestamp: datetime, equity: float):
        """
        Record equity snapshot.
        
        Args:
            timestamp: When equity was recorded
            equity: Current account equity
        """
        self.equity_history.append((timestamp, equity))
    
    def calculate_drawdown_metrics(self) -> Dict:
        """
        Calculate drawdown metrics from equity history.
        
        Returns:
            Dictionary with drawdown analysis
        """
        if not self.equity_history:
            return {
                'status': 'no_data',
                'message': 'No equity records to analyze'
            }
        
        # Sort by timestamp
        sorted_history = sorted(self.equity_history, key=lambda x: x[0])
        
        # Calculate running metrics
        peak_equity = self.initial_equity
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        current_drawdown = 0.0
        current_drawdown_pct = 0.0
        
        daily_drawdowns = defaultdict(lambda: {'min_equity': float('inf'), 'start_equity': 0})
        
        for i, (ts, equity) in enumerate(sorted_history):
            # Update peak
            if equity > peak_equity:
                peak_equity = equity
            
            # Calculate current drawdown
            current_drawdown = peak_equity - equity
            current_drawdown_pct = (current_drawdown / peak_equity) * 100 if peak_equity > 0 else 0
            
            # Update max
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                max_drawdown_pct = current_drawdown_pct
            
            # Track daily drawdowns
            day = ts.date()
            if i == 0 or sorted_history[i-1][0].date() != day:
                daily_drawdowns[day]['start_equity'] = equity
            daily_drawdowns[day]['min_equity'] = min(daily_drawdowns[day]['min_equity'], equity)
        
        # Calculate daily drawdown percentages
        daily_stats = {}
        for day, data in daily_drawdowns.items():
            start = data['start_equity']
            min_eq = data['min_equity']
            if start > 0:
                dd_pct = ((start - min_eq) / start) * 100
                daily_stats[str(day)] = {
                    'start_equity': f"${start:.2f}",
                    'min_equity': f"${min_eq:.2f}",
                    'drawdown_pct': f"{dd_pct:.2f}%"
                }
        
        # Get current state (last entry)
        last_ts, last_equity = sorted_history[-1]
        recovery = last_equity - (peak_equity - max_drawdown)
        recovery_pct = (recovery / max_drawdown) * 100 if max_drawdown > 0 else 100.0
        
        # Calculate return from start
        total_return = ((last_equity - self.initial_equity) / self.initial_equity) * 100
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'starting_equity': f"${self.initial_equity:.2f}",
            'current_equity': f"${last_equity:.2f}",
            'total_return': f"{total_return:.2f}%",
            'peak_equity': f"${peak_equity:.2f}",
            'max_drawdown': f"${max_drawdown:.2f}",
            'max_drawdown_pct': f"{max_drawdown_pct:.2f}%",
            'current_drawdown': f"${current_drawdown:.2f}",
            'current_drawdown_pct': f"{current_drawdown_pct:.2f}%",
            'recovery_from_max_dd': f"{recovery_pct:.2f}%",
            'daily_drawdowns': daily_stats,
            'equity_samples': len(self.equity_history),
        }
    
    def check_drawdown_limits(self, 
                            max_daily_dd: float = 0.05,
                            max_total_dd: float = 0.20) -> Dict[str, any]:
        """
        Check if drawdown limits are exceeded.
        
        Args:
            max_daily_dd: Maximum daily drawdown percentage (default 5%)
            max_total_dd: Maximum total drawdown percentage (default 20%)
        
        Returns:
            Dictionary with limit status and warnings
        """
        metrics = self.calculate_drawdown_metrics()
        
        if metrics.get('status') == 'no_data':
            return {'status': 'no_data'}
        
        # Extract percentages
        max_dd_pct = float(metrics['max_drawdown_pct'].strip('%'))
        warnings = []
        
        # Check maximum total drawdown
        if max_dd_pct > max_total_dd * 100:
            warnings.append(
                f"⚠️  Maximum drawdown exceeds limit: "
                f"{max_dd_pct:.2f}% > {max_total_dd*100:.1f}%"
            )
        
        # Check daily drawdowns
        max_daily_pct = 0.0
        for day, stats in metrics.get('daily_drawdowns', {}).items():
            daily_pct = float(stats['drawdown_pct'].strip('%'))
            max_daily_pct = max(max_daily_pct, daily_pct)
            
            if daily_pct > max_daily_dd * 100:
                warnings.append(
                    f"⚠️  Daily drawdown on {day} exceeds limit: "
                    f"{daily_pct:.2f}% > {max_daily_dd*100:.1f}%"
                )
        
        return {
            'status': 'within_limits' if not warnings else 'exceeded',
            'max_drawdown_pct': max_dd_pct,
            'max_daily_drawdown_pct': max_daily_pct,
            'max_total_limit': max_total_dd * 100,
            'max_daily_limit': max_daily_dd * 100,
            'warnings': warnings,
        }
    
    def generate_report(self, 
                       max_daily_dd: float = 0.05,
                       max_total_dd: float = 0.20) -> str:
        """
        Generate human-readable drawdown report.
        
        Args:
            max_daily_dd: Maximum daily drawdown limit
            max_total_dd: Maximum total drawdown limit
        
        Returns:
            Formatted report string
        """
        metrics = self.calculate_drawdown_metrics()
        limits = self.check_drawdown_limits(max_daily_dd, max_total_dd)
        
        if metrics.get('status') == 'no_data':
            return "No equity data to analyze yet. Check back after trading begins."
        
        report = []
        report.append("=" * 80)
        report.append("DRAWDOWN ANALYSIS")
        report.append("=" * 80)
        report.append(f"Report generated: {metrics['timestamp']}")
        report.append("")
        
        # Account equity
        report.append("ACCOUNT EQUITY")
        report.append("-" * 40)
        report.append(f"Starting equity:      {metrics['starting_equity']}")
        report.append(f"Current equity:       {metrics['current_equity']}")
        report.append(f"Peak equity:          {metrics['peak_equity']}")
        report.append(f"Total return:         {metrics['total_return']}")
        report.append("")
        
        # Drawdown metrics
        report.append("DRAWDOWN METRICS")
        report.append("-" * 40)
        report.append(f"Maximum drawdown:     {metrics['max_drawdown']} ({metrics['max_drawdown_pct']})")
        report.append(f"Current drawdown:     {metrics['current_drawdown']} ({metrics['current_drawdown_pct']})")
        report.append(f"Recovery from max DD: {limits['recovery_from_max_dd'] if 'recovery_from_max_dd' in limits else metrics.get('recovery_from_max_dd', 'N/A')}")
        report.append("")
        
        # Daily drawdowns
        if metrics.get('daily_drawdowns'):
            report.append("DAILY DRAWDOWNS (Last 10 days)")
            report.append("-" * 40)
            for day in sorted(metrics['daily_drawdowns'].keys())[-10:]:
                stats = metrics['daily_drawdowns'][day]
                report.append(f"{day} | Start: {stats['start_equity']:>11} | "
                            f"Min: {stats['min_equity']:>11} | DD: {stats['drawdown_pct']:>7}")
            report.append("")
        
        # Limit checks
        report.append("LIMIT CHECKS")
        report.append("-" * 40)
        report.append(f"Max total DD limit:   {max_total_dd*100:.1f}%")
        report.append(f"Current max DD:       {limits['max_drawdown_pct']:.2f}%")
        report.append(f"Status:               {'✓ PASSED' if limits['status'] == 'within_limits' else '✗ EXCEEDED'}")
        report.append("")
        report.append(f"Max daily DD limit:   {max_daily_dd*100:.1f}%")
        report.append(f"Max observed daily DD: {limits['max_daily_drawdown_pct']:.2f}%")
        report.append(f"Status:               {'✓ PASSED' if not any('daily' in w for w in limits.get('warnings', [])) else '✗ EXCEEDED'}")
        report.append("")
        
        # Warnings
        if limits.get('warnings'):
            report.append("ALERTS")
            report.append("-" * 40)
            for warning in limits['warnings']:
                report.append(f"  {warning}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Monitor drawdown metrics from trading logs'
    )
    parser.add_argument(
        '--initial-equity',
        type=float,
        default=100000.0,
        help='Starting account equity'
    )
    parser.add_argument(
        '--max-daily-dd',
        type=float,
        default=0.05,
        help='Maximum daily drawdown (0.05 = 5%)'
    )
    parser.add_argument(
        '--max-total-dd',
        type=float,
        default=0.20,
        help='Maximum total drawdown (0.20 = 20%)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    monitor = DrawdownMonitor(initial_equity=args.initial_equity)
    
    # Note: In real usage, would load equity history from actual trades/logs
    # This is a template showing how to use the monitor
    
    if args.json:
        metrics = monitor.calculate_drawdown_metrics()
        print(json.dumps(metrics, indent=2))
    else:
        report = monitor.generate_report(
            max_daily_dd=args.max_daily_dd,
            max_total_dd=args.max_total_dd
        )
        print(report)


if __name__ == '__main__':
    main()

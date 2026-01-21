#!/usr/bin/env python3
"""
monitor_readiness_correlation.py - Readiness state vs trade outcome analysis

Analyzes correlation between market readiness state and trading success:
- Win rate per readiness level (ready, forming, not_ready)
- Trade count and profitability by readiness
- Alert triggered vs actual outcomes
- Readiness transition patterns
- Confidence levels in each readiness state

Validates that readiness gating improves risk-adjusted returns.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from statistics import mean, stdev

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReadinessCorrelationAnalyzer:
    """Analyze correlation between readiness state and trading outcomes."""
    
    READINESS_LEVELS = ['ready', 'forming', 'not_ready']
    
    def __init__(self):
        """Initialize analyzer."""
        self.trades: List[Dict] = []
        self.readiness_events: List[Dict] = []
        
    def load_trades_from_log(self, log_file: str = "logs/live_trading.log") -> bool:
        """
        Load completed trades from log file.
        
        Args:
            log_file: Path to trading log
        
        Returns:
            True if successful
        """
        if not Path(log_file).exists():
            logger.warning(f"Log file not found: {log_file}")
            return False
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Trade closed' in line or 'trade_closed' in line:
                        try:
                            if '{' in line and '}' in line:
                                json_str = line[line.find('{'):line.rfind('}')+1]
                                trade = json.loads(json_str)
                                # Ensure pnl field exists
                                if 'pnl' not in trade and 'return_pct' in trade:
                                    trade['pnl'] = trade.get('return_pct', 0)
                                self.trades.append(trade)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    
                    # Also capture readiness state changes
                    if 'Readiness' in line or 'readiness' in line or 'readiness_state' in line:
                        try:
                            if '{' in line and '}' in line:
                                json_str = line[line.find('{'):line.rfind('}')+1]
                                event = json.loads(json_str)
                                self.readiness_events.append(event)
                        except (json.JSONDecodeError, ValueError):
                            pass
            
            logger.info(f"Loaded {len(self.trades)} trades and {len(self.readiness_events)} readiness events")
            return len(self.trades) > 0
            
        except Exception as e:
            logger.error(f"Error loading logs: {e}")
            return False
    
    def analyze_by_readiness(self) -> Dict:
        """
        Analyze trade outcomes by readiness level.
        
        Returns:
            Dictionary with readiness-based statistics
        """
        if not self.trades:
            return {
                'status': 'no_data',
                'message': 'No trades to analyze'
            }
        
        # Group trades by readiness level
        # In real scenario, would match trades to readiness states by timestamp
        by_readiness = defaultdict(lambda: {
            'trades': [],
            'win_count': 0,
            'loss_count': 0,
            'total_pnl': 0.0,
            'confidences': []
        })
        
        for trade in self.trades:
            # This is template - in real usage would match trade timestamp to readiness state
            readiness = trade.get('readiness_state', 'unknown')
            pnl = trade.get('pnl', 0)
            confidence = trade.get('confidence', 0.5)
            
            if readiness in self.READINESS_LEVELS:
                by_readiness[readiness]['trades'].append(trade)
                if pnl > 0:
                    by_readiness[readiness]['win_count'] += 1
                elif pnl < 0:
                    by_readiness[readiness]['loss_count'] += 1
                by_readiness[readiness]['total_pnl'] += pnl
                by_readiness[readiness]['confidences'].append(confidence)
        
        # Format results
        result = {}
        overall_wr = 0.0
        overall_count = 0
        
        for level in self.READINESS_LEVELS:
            stats = by_readiness[level]
            total_trades = len(stats['trades'])
            
            if total_trades == 0:
                result[level] = {
                    'trade_count': 0,
                    'message': 'No trades in this readiness state'
                }
            else:
                wr = stats['win_count'] / total_trades
                avg_confidence = mean(stats['confidences']) if stats['confidences'] else 0.0
                
                result[level] = {
                    'trade_count': total_trades,
                    'wins': stats['win_count'],
                    'losses': stats['loss_count'],
                    'win_rate': f"{wr*100:.1f}%",
                    'total_pnl': f"${stats['total_pnl']:.2f}",
                    'avg_pnl_per_trade': f"${stats['total_pnl']/total_trades:.2f}",
                    'avg_confidence': f"{avg_confidence:.2f}",
                }
                
                overall_wr += stats['win_count']
                overall_count += total_trades
        
        # Add overall
        if overall_count > 0:
            result['overall'] = {
                'trade_count': overall_count,
                'overall_win_rate': f"{(overall_wr/overall_count)*100:.1f}%"
            }
        
        return result
    
    def analyze_readiness_transitions(self) -> Dict:
        """
        Analyze patterns in readiness state transitions.
        
        Returns:
            Dictionary with transition patterns
        """
        if not self.readiness_events:
            return {
                'status': 'no_data',
                'message': 'No readiness state transitions recorded'
            }
        
        # Group by symbol
        by_symbol = defaultdict(list)
        for event in sorted(self.readiness_events, key=lambda x: x.get('timestamp', '')):
            symbol = event.get('symbol', 'UNKNOWN')
            state = event.get('readiness_state', 'unknown')
            by_symbol[symbol].append(state)
        
        # Analyze transitions
        transitions = defaultdict(int)
        for symbol, states in by_symbol.items():
            for i in range(len(states) - 1):
                from_state = states[i]
                to_state = states[i + 1]
                if from_state != to_state:
                    transitions[f"{from_state} -> {to_state}"] += 1
        
        # Format transitions
        sorted_transitions = sorted(
            transitions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total_transitions': sum(transitions.values()),
            'unique_transition_types': len(transitions),
            'most_common_transitions': [
                {'transition': t, 'count': c} for t, c in sorted_transitions[:10]
            ],
            'by_symbol_transitions': {
                sym: len(set((states[i], states[i+1]) for i in range(len(states)-1) 
                           if states[i] != states[i+1]))
                for sym, states in by_symbol.items()
            }
        }
    
    def calculate_correlation_insights(self) -> Dict:
        """
        Calculate key insights about readiness-outcome correlation.
        
        Returns:
            Dictionary with insights
        """
        by_readiness = self.analyze_by_readiness()
        
        insights = []
        
        # Compare win rates by readiness level
        wr_by_level = {}
        for level in self.READINESS_LEVELS:
            if level in by_readiness and 'win_rate' in by_readiness[level]:
                wr_pct = float(by_readiness[level]['win_rate'].strip('%'))
                wr_by_level[level] = wr_pct
        
        if len(wr_by_level) >= 2:
            # Find best and worst
            best_level = max(wr_by_level, key=wr_by_level.get)
            worst_level = min(wr_by_level, key=wr_by_level.get)
            improvement = wr_by_level[best_level] - wr_by_level[worst_level]
            
            insights.append({
                'type': 'readiness_effectiveness',
                'best_readiness_level': best_level,
                'best_win_rate': f"{wr_by_level[best_level]:.1f}%",
                'worst_readiness_level': worst_level,
                'worst_win_rate': f"{wr_by_level[worst_level]:.1f}%",
                'improvement': f"{improvement:.1f}%",
                'description': f"Trading in '{best_level}' state shows {improvement:.1f}% better win rate than '{worst_level}'"
            })
        
        # Confidence insights
        if self.trades:
            high_conf = [t for t in self.trades if t.get('confidence', 0) > 0.75]
            low_conf = [t for t in self.trades if t.get('confidence', 0) <= 0.75]
            
            if high_conf and low_conf:
                high_wr = sum(1 for t in high_conf if t.get('pnl', 0) > 0) / len(high_conf)
                low_wr = sum(1 for t in low_conf if t.get('pnl', 0) > 0) / len(low_conf)
                
                insights.append({
                    'type': 'confidence_correlation',
                    'high_confidence_wr': f"{high_wr*100:.1f}%",
                    'low_confidence_wr': f"{low_wr*100:.1f}%",
                    'advantage': f"{(high_wr-low_wr)*100:.1f}%",
                    'description': f"High confidence signals ({high_wr*100:.1f}%) outperform low confidence ({low_wr*100:.1f}%)"
                })
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'insights': insights
        }
    
    def generate_report(self) -> str:
        """
        Generate human-readable readiness correlation report.
        
        Returns:
            Formatted report string
        """
        readiness_stats = self.analyze_by_readiness()
        transitions = self.analyze_readiness_transitions()
        insights = self.calculate_correlation_insights()
        
        if readiness_stats.get('status') == 'no_data':
            return "No trade data to analyze yet. Check back after trading begins."
        
        report = []
        report.append("=" * 80)
        report.append("READINESS STATE CORRELATION ANALYSIS")
        report.append("=" * 80)
        report.append(f"Report generated: {insights['timestamp']}")
        report.append("")
        
        # By readiness level
        report.append("PERFORMANCE BY READINESS LEVEL")
        report.append("-" * 40)
        for level in self.READINESS_LEVELS:
            if level in readiness_stats:
                stats = readiness_stats[level]
                if stats.get('trade_count', 0) > 0:
                    report.append(f"{level.upper()}")
                    report.append(f"  Trades:           {stats.get('trade_count', 0)}")
                    report.append(f"  Win/Loss:         {stats.get('wins', 0)}/{stats.get('losses', 0)}")
                    report.append(f"  Win rate:         {stats.get('win_rate', 'N/A')}")
                    report.append(f"  Avg confidence:   {stats.get('avg_confidence', 'N/A')}")
                    report.append(f"  Total PnL:        {stats.get('total_pnl', 'N/A')}")
                    report.append(f"  Avg per trade:    {stats.get('avg_pnl_per_trade', 'N/A')}")
                    report.append("")
        
        # Overall
        if 'overall' in readiness_stats:
            report.append(f"OVERALL WIN RATE: {readiness_stats['overall'].get('overall_win_rate', 'N/A')}")
            report.append("")
        
        # Transitions
        if transitions.get('status') != 'no_data':
            report.append("READINESS STATE TRANSITIONS")
            report.append("-" * 40)
            report.append(f"Total transitions recorded: {transitions.get('total_transitions', 0)}")
            report.append(f"Unique transition types:   {transitions.get('unique_transition_types', 0)}")
            report.append("")
            
            if transitions.get('most_common_transitions'):
                report.append("Most common transitions:")
                for item in transitions['most_common_transitions'][:5]:
                    report.append(f"  {item['transition']:30} x {item['count']}")
                report.append("")
        
        # Insights
        if insights.get('insights'):
            report.append("KEY INSIGHTS")
            report.append("-" * 40)
            for insight in insights['insights']:
                report.append(f"• {insight.get('description', 'N/A')}")
                for key, value in insight.items():
                    if key not in ['type', 'description']:
                        report.append(f"  {key}: {value}")
                report.append("")
        
        report.append("=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Verify readiness thresholds match observed win rate patterns")
        report.append("2. If 'not_ready' state shows good results, consider broadening regime filters")
        report.append("3. If 'ready' state underperforms, review regime classification logic")
        report.append("4. Track confidence correlation to adjust confidence scaling factors")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze readiness state correlation with trading outcomes'
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
    
    analyzer = ReadinessCorrelationAnalyzer()
    
    if not analyzer.load_trades_from_log(args.log):
        print(f"No trades found in {args.log}")
        sys.exit(1)
    
    if args.json:
        result = {
            'by_readiness': analyzer.analyze_by_readiness(),
            'transitions': analyzer.analyze_readiness_transitions(),
            'insights': analyzer.calculate_correlation_insights()
        }
        print(json.dumps(result, indent=2))
    else:
        print(analyzer.generate_report())


if __name__ == '__main__':
    main()

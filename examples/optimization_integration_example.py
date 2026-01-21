#!/usr/bin/env python3
"""
example_optimization_integration.py - Example of integrating optimization with live trading

This example shows how to integrate the optimization system with your live trading setup.
Copy and adapt to your actual trading code.

Key integration points:
1. Initialize optimization at startup
2. Score signals before execution
3. Apply position size multiplier
4. Periodic optimization updates
5. Monitor and alert
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Example 1: Initialize Optimization at Startup
# ==============================================================================

def initialize_optimization():
    """Initialize the optimization system at startup."""
    from ai.optimization_engine import initialize_engine
    from ai.performance_decomposer import record_trade
    from ai.optimization_monitor import get_monitor
    
    # Define base strategy weights
    base_weights = {
        "EMA_Trend": 1.0,      # Trend-following strategy
        "FVG_Pattern": 1.0,    # Fair Value Gap pattern strategy
        "RSI_Reversal": 0.8,   # Reversal strategy (given less weight)
    }
    
    # Initialize optimization engine with conservative settings
    engine = initialize_engine(
        base_weights=base_weights,
        risk_max_drawdown=0.20,  # Never exceed 20% portfolio drawdown
        risk_max_position_size=0.05,  # Max 5% per position
        enable_quality_filtering=True,  # Filter low-quality signals
        enable_adaptive_weighting=True,  # Adjust weights based on performance
        enable_retirement_management=True,  # Auto-suspend underperformers
    )
    
    # Initialize monitoring
    monitor = get_monitor()
    
    logger.info("Optimization system initialized successfully")
    
    return engine, monitor


# ==============================================================================
# Example 2: Score and Filter Signals Before Execution
# ==============================================================================

def generate_and_filter_signals(
    strategy_name: str,
    symbol: str,
    signal_type: str,
    original_confidence: float,
    confluence_count: int = 1,
    regime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate trading signal and apply optimization filters.
    
    This should be called for every potential trade before execution.
    
    Args:
        strategy_name: Name of strategy (e.g., "EMA_Trend")
        symbol: Trading pair (e.g., "BTC/USDT")
        signal_type: "BUY", "SELL", or "HOLD"
        original_confidence: Original confidence from strategy (0.0-1.0)
        confluence_count: How many indicators agree (1-5+)
        regime: Current market regime ("TRENDING", "RANGING", "VOLATILE")
    
    Returns:
        Dict with: {
            'signal': 'BUY'|'SELL'|'HOLD',
            'quality': float (0.0-1.0),
            'should_execute': bool,
            'multiplier': float (0.0-2.0),
        }
    """
    from ai.optimization_engine import get_engine
    
    engine = get_engine()
    
    # Score the signal
    quality, should_execute = engine.score_signal(
        strategy=strategy_name,
        symbol=symbol,
        signal_type=signal_type,
        original_confidence=original_confidence,
        confluence_count=confluence_count,
        regime=regime,
    )
    
    if not should_execute:
        logger.info(
            f"Signal filtered: {strategy_name}/{symbol} {signal_type} "
            f"quality={quality:.3f} (threshold={engine._quality_scorer.min_quality_threshold:.2f})"
        )
        return {
            'signal': 'HOLD',
            'quality': quality,
            'should_execute': False,
            'multiplier': 0.0,
            'reason': 'Quality threshold not met',
        }
    
    # Get position size multiplier
    multiplier = engine.get_position_size_multiplier(
        strategy=strategy_name,
        signal_quality=quality,
        regime=regime,
    )
    
    logger.info(
        f"Signal approved: {strategy_name}/{symbol} {signal_type} "
        f"quality={quality:.3f} multiplier={multiplier:.2f}x"
    )
    
    return {
        'signal': signal_type,
        'quality': quality,
        'should_execute': True,
        'multiplier': multiplier,
    }


# ==============================================================================
# Example 3: Execute Trade with Optimization
# ==============================================================================

def execute_trade_with_optimization(
    strategy_name: str,
    symbol: str,
    signal_info: Dict[str, Any],
    base_position_size: float,
) -> Optional[str]:
    """
    Execute a trade with optimization multiplier applied.
    
    Args:
        strategy_name: Strategy name
        symbol: Trading pair
        signal_info: Output from generate_and_filter_signals()
        base_position_size: Base position size (from risk module)
    
    Returns:
        Order ID if executed, None otherwise
    """
    
    if not signal_info.get('should_execute'):
        logger.info(f"Trade not executed: {signal_info.get('reason')}")
        return None
    
    # Apply optimization multiplier to position size
    final_position_size = base_position_size * signal_info['multiplier']
    
    logger.info(
        f"Executing {signal_info['signal']}: {symbol} "
        f"size={final_position_size:.6f} ({signal_info['multiplier']:.2f}x multiplier)"
    )
    
    # Call your actual execution function here
    # order_id = place_order(symbol, signal_info['signal'], final_position_size)
    order_id = f"ORDER_{strategy_name}_{datetime.now().timestamp()}"
    
    return order_id


# ==============================================================================
# Example 4: Record Trade and Update Performance
# ==============================================================================

def record_completed_trade(
    strategy_name: str,
    symbol: str,
    pnl: float,
    entry_timestamp_ms: int,
    exit_timestamp_ms: int,
    regime: Optional[str] = None,
    size: Optional[float] = None,
):
    """
    Record a completed trade for performance tracking.
    
    This should be called after a trade closes, before the next optimization update.
    
    Args:
        strategy_name: Strategy that generated the trade
        symbol: Trading pair
        pnl: Realized profit/loss
        entry_timestamp_ms: Entry time (milliseconds since epoch)
        exit_timestamp_ms: Exit time (milliseconds since epoch)
        regime: Market regime when trade was closed
        size: Trade size
    """
    from ai.performance_decomposer import record_trade
    
    record_trade(
        strategy_name=strategy_name,
        symbol=symbol,
        pnl=pnl,
        entry_ts=entry_timestamp_ms,
        exit_ts=exit_timestamp_ms,
        regime=regime,
        size=size,
    )
    
    logger.info(
        f"Trade recorded: {strategy_name}/{symbol} PnL={pnl:.2f} "
        f"regime={regime} size={size}"
    )


# ==============================================================================
# Example 5: Periodic Optimization Update
# ==============================================================================

async def optimization_update_loop(
    update_interval_seconds: int = 3600,  # Every hour
):
    """
    Periodically update all optimization components.
    
    This should run in the background, separate from trade execution.
    Recommended: Update every 1-24 hours depending on trading frequency.
    
    Args:
        update_interval_seconds: Update interval (default: 3600 = 1 hour)
    """
    from ai.optimization_engine import get_engine
    from ai.optimization_monitor import get_monitor
    
    engine = get_engine()
    monitor = get_monitor()
    
    while True:
        try:
            await asyncio.sleep(update_interval_seconds)
            
            # Get current market regime
            regime = detect_market_regime()  # Implement this with your regime detector
            
            # Update all optimization components
            state = engine.update_optimization(regime=regime)
            
            # Log significant changes
            if state.weight_changes_this_period:
                logger.info(f"Weight updates: {state.weight_changes_this_period}")
            
            if state.degraded_strategies:
                logger.warning(f"Degraded strategies: {state.degraded_strategies}")
            
            if state.suspended_strategies:
                logger.critical(f"Suspended strategies: {state.suspended_strategies}")
            
            # Generate periodic report
            health = monitor.get_health_report()
            logger.info(
                f"Strategy health: {len(health['active'])} active, "
                f"{len(health['degraded'])} degraded, "
                f"{len(health['suspended'])} suspended"
            )
            
        except Exception as e:
            logger.error(f"Optimization update failed: {e}", exc_info=True)


# ==============================================================================
# Example 6: Monitoring and Alerting
# ==============================================================================

def generate_optimization_alerts() -> list:
    """
    Generate alerts based on optimization state.
    
    Can be called periodically or on-demand for integration with your
    alerting system (Slack, Discord, email, etc.).
    
    Returns:
        List of alert strings
    """
    from ai.optimization_monitor import get_monitor
    
    monitor = get_monitor()
    alerts = []
    
    # Get overall status
    health = monitor.get_health_report()
    
    # Alert on critical issues
    if not health['active']:
        alerts.append("CRITICAL: No active strategies!")
    
    # Alert on suspended strategies
    if health['suspended']:
        alerts.append(f"WARNING: Suspended strategies: {', '.join(health['suspended'])}")
    
    # Alert on multiple degraded
    if len(health['degraded']) > 2:
        alerts.append(f"ALERT: {len(health['degraded'])} strategies degraded")
    
    # Alert on allocation imbalance
    alloc = monitor.get_allocation_report()
    normalized = alloc['normalized_weights']
    if normalized:
        max_weight = max(normalized.values())
        if max_weight > 0.6:  # One strategy > 60%
            alerts.append(f"ALERT: Allocation imbalance (max weight: {max_weight:.1%})")
    
    # Alert on signal quality issues
    opt_status = monitor.get_optimization_status()
    if opt_status['status'] == 'error':
        alerts.append(f"ERROR: Optimization status check failed")
    
    return alerts


def send_alerts(alerts: list):
    """Send alerts to your notification system."""
    for alert in alerts:
        logger.warning(alert)
        # Integrate with your alerting: Slack, Discord, email, etc.
        # send_slack_message(alert)
        # send_discord_message(alert)


# ==============================================================================
# Example 7: Generate Reports
# ==============================================================================

def generate_daily_optimization_report():
    """
    Generate daily optimization performance report.
    
    Can be scheduled to run daily (e.g., via cron or scheduler).
    """
    from ai.optimization_monitor import get_monitor
    from datetime import datetime, timezone
    
    monitor = get_monitor()
    
    # Gather all reports
    optimization_status = monitor.get_optimization_status()
    health_report = monitor.get_health_report()
    allocation_report = monitor.get_allocation_report()
    regime_effectiveness = monitor.get_regime_effectiveness_report()
    
    # Create report
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'optimization': optimization_status,
        'strategy_health': health_report,
        'allocation': allocation_report,
        'regime_effectiveness': regime_effectiveness,
    }
    
    # Export to file
    import json
    filename = f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report exported to {filename}")
    
    return report


# ==============================================================================
# Example 8: Helper Function (Placeholder)
# ==============================================================================

def detect_market_regime() -> str:
    """
    Detect current market regime.
    
    This is a placeholder - integrate with your actual regime detector.
    
    Returns:
        Regime string: "TRENDING", "RANGING", "VOLATILE", or "QUIET"
    """
    # TODO: Implement with your RegimeDetector
    # from strategies.regime_detector import RegimeDetector
    # detector = RegimeDetector()
    # regime = detector.detect_regime(market_data)
    
    return "TRENDING"  # Default for now


# ==============================================================================
# Example: Complete Integration
# ==============================================================================

async def main():
    """
    Complete example workflow.
    """
    
    # 1. Initialize
    logger.info("=" * 60)
    logger.info("Starting optimization system example")
    logger.info("=" * 60)
    
    engine, monitor = initialize_optimization()
    
    # 2. Simulate some trades for context
    from ai.performance_decomposer import record_trade
    
    record_trade("EMA_Trend", "BTC/USDT", 100.0, 1704067200000, 1704153600000, "TRENDING")
    record_trade("EMA_Trend", "BTC/USDT", 50.0, 1704153600000, 1704240000000, "TRENDING")
    record_trade("FVG_Pattern", "BTC/USDT", -30.0, 1704067200000, 1704153600000, "RANGING")
    
    logger.info("\nTrades recorded. Generating signals...")
    
    # 3. Generate signals and filter
    signal_btc = generate_and_filter_signals(
        strategy_name="EMA_Trend",
        symbol="BTC/USDT",
        signal_type="BUY",
        original_confidence=0.8,
        confluence_count=3,
        regime="TRENDING",
    )
    
    signal_eth = generate_and_filter_signals(
        strategy_name="FVG_Pattern",
        symbol="ETH/USDT",
        signal_type="SELL",
        original_confidence=0.4,
        confluence_count=1,
        regime="RANGING",
    )
    
    # 4. Execute trades
    if signal_btc['should_execute']:
        order_id = execute_trade_with_optimization(
            "EMA_Trend", "BTC/USDT", signal_btc, base_position_size=0.5
        )
    
    if signal_eth['should_execute']:
        order_id = execute_trade_with_optimization(
            "FVG_Pattern", "ETH/USDT", signal_eth, base_position_size=0.3
        )
    
    # 5. Generate reports
    logger.info("\nGenerating optimization reports...")
    health = monitor.get_health_report()
    alloc = monitor.get_allocation_report()
    
    logger.info(f"\nStrategy Health:")
    logger.info(f"  Active: {health['active']}")
    logger.info(f"  Degraded: {health['degraded']}")
    logger.info(f"  Suspended: {health['suspended']}")
    
    logger.info(f"\nAllocation:")
    logger.info(f"  Weights: {alloc['raw_weights']}")
    logger.info(f"  Normalized: {alloc['normalized_weights']}")
    
    # 6. Check for alerts
    logger.info("\nChecking for alerts...")
    alerts = generate_optimization_alerts()
    if alerts:
        logger.warning("Alerts found:")
        for alert in alerts:
            logger.warning(f"  - {alert}")
    else:
        logger.info("No alerts")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example complete. Optimization system is ready for live trading.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

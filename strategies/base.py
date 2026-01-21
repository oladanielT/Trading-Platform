"""
base.py - Abstract Base Class for Trading Strategies

Defines the contract that all trading strategies must implement.
Strategies generate signals based on market data and position state,
but do not execute trades directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

try:
    import pandas as pd
except ImportError:
    pd = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"  # Close existing position
    
    def __str__(self):
        return self.value


@dataclass
class SignalOutput:
    """
    Standardized signal output from strategy.
    
    Attributes:
        signal: Trading signal (BUY/SELL/HOLD/CLOSE)
        confidence: Signal confidence level (0.0 to 1.0)
        metadata: Additional signal information
        timestamp: Time signal was generated
        strategy_name: Name of strategy that generated signal
    """
    signal: Signal
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    strategy_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate signal output after initialization."""
        # Validate confidence is in valid range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        # Ensure signal is Signal enum
        if isinstance(self.signal, str):
            self.signal = Signal[self.signal.upper()]
        elif not isinstance(self.signal, Signal):
            raise ValueError(f"Signal must be Signal enum or string, got {type(self.signal)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert signal output to dictionary.
        
        Returns:
            Dictionary representation of signal
        """
        return {
            'signal': self.signal.value,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'strategy_name': self.strategy_name
        }
    
    def is_actionable(self, min_confidence: float = 0.0) -> bool:
        """
        Check if signal is actionable (not HOLD and meets confidence threshold).
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if signal is actionable
        """
        return self.signal != Signal.HOLD and self.confidence >= min_confidence
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SignalOutput(signal={self.signal.value}, "
                f"confidence={self.confidence:.2f}, "
                f"strategy={self.strategy_name})")


@dataclass
class PositionState:
    """
    Current position state information.
    
    Attributes:
        has_position: Whether currently holding a position
        position_type: Type of position ('long', 'short', or None)
        entry_price: Price at which position was entered
        position_size: Size of current position
        unrealized_pnl: Unrealized profit/loss
        entry_time: Time position was entered
        metadata: Additional position information
    """
    has_position: bool = False
    position_type: Optional[str] = None
    entry_price: Optional[float] = None
    position_size: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    entry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'has_position': self.has_position,
            'position_type': self.position_type,
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'unrealized_pnl': self.unrealized_pnl,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionState':
        """Create PositionState from dictionary."""
        if 'entry_time' in data and isinstance(data['entry_time'], str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation."""
        if not self.has_position:
            return "PositionState(has_position=False)"
        return (f"PositionState(position_type={self.position_type}, "
                f"entry_price={self.entry_price}, "
                f"size={self.position_size})")


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement the generate_signal method which produces
    trading signals based on market data and current position state.
    Strategies do not execute trades - they only generate signals.
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name (defaults to class name)
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name or self.__class__.__name__
        self.parameters = kwargs
        self._initialized = True
        
        logger.info(f"Strategy '{self.name}' initialized with parameters: {kwargs}")
    
    @abstractmethod
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        position_state: PositionState
    ) -> SignalOutput:
        """
        Generate trading signal based on market data and position state.
        
        This is the core method that must be implemented by all strategies.
        It analyzes market data and current position to produce a signal.
        
        Args:
            market_data: DataFrame with OHLCV data and any indicators
                        Must have columns: timestamp, open, high, low, close, volume
            position_state: Current position state information
            
        Returns:
            SignalOutput with signal, confidence, and metadata
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Strategy must implement generate_signal method")
    
    def validate_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Validate that market data has required columns.
        
        Args:
            market_data: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        if pd is None:
            raise ImportError("pandas required. Install with: pip install pandas")
        
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError(f"market_data must be pandas DataFrame, got {type(market_data)}")
        
        if market_data.empty:
            raise ValueError("market_data is empty")
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        
        if missing_columns:
            raise ValueError(f"market_data missing required columns: {missing_columns}")
    
    def get_latest_price(self, market_data: pd.DataFrame) -> float:
        """
        Get the latest close price from market data.
        
        Args:
            market_data: DataFrame with price data
            
        Returns:
            Latest close price
        """
        if market_data.empty:
            raise ValueError("market_data is empty")
        return float(market_data['close'].iloc[-1])
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get strategy parameter value.
        
        Args:
            key: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set strategy parameter value.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        self.parameters[key] = value
        logger.debug(f"Strategy '{self.name}' parameter updated: {key}={value}")
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Get all strategy parameters.
        
        Returns:
            Dictionary of all parameters
        """
        return self.parameters.copy()
    
    def create_signal(
        self,
        signal: Signal,
        confidence: float,
        **metadata
    ) -> SignalOutput:
        """
        Helper method to create a SignalOutput with strategy name.
        
        Args:
            signal: Signal type
            confidence: Signal confidence (0.0-1.0)
            **metadata: Additional metadata as keyword arguments
            
        Returns:
            SignalOutput instance
        """
        return SignalOutput(
            signal=signal,
            confidence=confidence,
            metadata=metadata,
            strategy_name=self.name
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}(parameters={self.parameters})"


class CombinedStrategy(BaseStrategy):
    """
    Strategy that combines multiple sub-strategies.
    
    Aggregates signals from multiple strategies using a specified combination method.
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        combination_method: str = 'majority',
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize combined strategy.
        
        Args:
            strategies: List of strategies to combine
            combination_method: How to combine signals ('majority', 'unanimous', 'weighted')
            name: Strategy name
            **kwargs: Additional parameters
        """
        super().__init__(name=name or "CombinedStrategy", **kwargs)
        
        if not strategies:
            raise ValueError("At least one strategy required")
        
        self.strategies = strategies
        self.combination_method = combination_method
        
        logger.info(
            f"CombinedStrategy initialized with {len(strategies)} strategies "
            f"using '{combination_method}' combination"
        )
    
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        position_state: PositionState
    ) -> SignalOutput:
        """
        Generate combined signal from all sub-strategies.
        
        Args:
            market_data: Market data DataFrame
            position_state: Current position state
            
        Returns:
            Combined SignalOutput
        """
        # Collect signals from all strategies
        signals = []
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(market_data, position_state)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not signals:
            return self.create_signal(Signal.HOLD, 0.0, error="All strategies failed")
        
        # Combine signals based on method
        if self.combination_method == 'majority':
            return self._combine_by_majority(signals)
        elif self.combination_method == 'unanimous':
            return self._combine_by_unanimity(signals)
        elif self.combination_method == 'weighted':
            return self._combine_by_weight(signals)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def _combine_by_majority(self, signals: List[SignalOutput]) -> SignalOutput:
        """Combine signals by majority vote."""
        signal_counts = {}
        for sig in signals:
            signal_counts[sig.signal] = signal_counts.get(sig.signal, 0) + 1
        
        majority_signal = max(signal_counts, key=signal_counts.get)
        avg_confidence = sum(s.confidence for s in signals if s.signal == majority_signal) / signal_counts[majority_signal]
        
        return self.create_signal(
            majority_signal,
            avg_confidence,
            method='majority',
            vote_counts=signal_counts,
            individual_signals=[s.to_dict() for s in signals]
        )
    
    def _combine_by_unanimity(self, signals: List[SignalOutput]) -> SignalOutput:
        """Combine signals requiring unanimous agreement."""
        first_signal = signals[0].signal
        
        if all(s.signal == first_signal for s in signals):
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            return self.create_signal(
                first_signal,
                avg_confidence,
                method='unanimous',
                agreement=True
            )
        else:
            return self.create_signal(
                Signal.HOLD,
                0.0,
                method='unanimous',
                agreement=False,
                individual_signals=[s.to_dict() for s in signals]
            )
    
    def _combine_by_weight(self, signals: List[SignalOutput]) -> SignalOutput:
        """Combine signals using confidence as weight."""
        weighted_votes = {}
        total_weight = 0.0
        
        for sig in signals:
            if sig.signal not in weighted_votes:
                weighted_votes[sig.signal] = 0.0
            weighted_votes[sig.signal] += sig.confidence
            total_weight += sig.confidence
        
        if total_weight == 0:
            return self.create_signal(Signal.HOLD, 0.0, method='weighted', error="Zero total weight")
        
        # Normalize and find highest weighted signal
        max_signal = max(weighted_votes, key=weighted_votes.get)
        max_weight = weighted_votes[max_signal]
        
        return self.create_signal(
            max_signal,
            max_weight / total_weight,
            method='weighted',
            weighted_votes=weighted_votes,
            individual_signals=[s.to_dict() for s in signals]
        )


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    print("=== Strategy Base Class Example ===\n")
    
    # Example strategy implementation
    class SimpleMAStrategy(BaseStrategy):
        """Example strategy using moving average crossover."""
        
        def __init__(self, fast_period: int = 10, slow_period: int = 20, **kwargs):
            super().__init__(
                name=kwargs.pop('name', 'SimpleMAStrategy'),
                fast_period=fast_period,
                slow_period=slow_period,
                **kwargs
            )
        
        def generate_signal(
            self,
            market_data: pd.DataFrame,
            position_state: PositionState
        ) -> SignalOutput:
            """Generate signal based on MA crossover."""
            # Validate data
            self.validate_market_data(market_data)
            
            # Calculate MAs
            fast_period = self.get_parameter('fast_period', 10)
            slow_period = self.get_parameter('slow_period', 20)
            
            if len(market_data) < slow_period:
                return self.create_signal(
                    Signal.HOLD,
                    0.0,
                    reason="Insufficient data"
                )
            
            fast_ma = market_data['close'].rolling(fast_period).mean()
            slow_ma = market_data['close'].rolling(slow_period).mean()
            
            # Get current and previous values
            current_fast = fast_ma.iloc[-1]
            current_slow = slow_ma.iloc[-1]
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]
            
            # Generate signal
            if prev_fast <= prev_slow and current_fast > current_slow:
                # Bullish crossover
                confidence = min(abs(current_fast - current_slow) / current_slow, 1.0)
                return self.create_signal(
                    Signal.BUY,
                    confidence,
                    reason="Bullish MA crossover",
                    fast_ma=current_fast,
                    slow_ma=current_slow
                )
            elif prev_fast >= prev_slow and current_fast < current_slow:
                # Bearish crossover
                confidence = min(abs(current_fast - current_slow) / current_slow, 1.0)
                return self.create_signal(
                    Signal.SELL,
                    confidence,
                    reason="Bearish MA crossover",
                    fast_ma=current_fast,
                    slow_ma=current_slow
                )
            else:
                return self.create_signal(
                    Signal.HOLD,
                    0.5,
                    reason="No crossover",
                    fast_ma=current_fast,
                    slow_ma=current_slow
                )
    
    try:
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(50) * 0.1,
            'high': prices + np.abs(np.random.randn(50) * 0.3),
            'low': prices - np.abs(np.random.randn(50) * 0.3),
            'close': prices,
            'volume': np.random.randint(1000, 5000, 50)
        })
        
        print("Sample market data:")
        print(market_data.tail())
        
        # Create position state
        position = PositionState(
            has_position=False
        )
        
        print(f"\nPosition state: {position}\n")
        
        # Create and test strategy
        print("1. Testing SimpleMAStrategy...")
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
        print(f"Strategy: {strategy}")
        print(f"Parameters: {strategy.get_all_parameters()}\n")
        
        signal = strategy.generate_signal(market_data, position)
        print(f"Generated signal: {signal}")
        print(f"Signal dict: {signal.to_dict()}")
        print(f"Is actionable: {signal.is_actionable(min_confidence=0.5)}\n")
        
        # Test combined strategy
        print("2. Testing CombinedStrategy...")
        strategy1 = SimpleMAStrategy(fast_period=5, slow_period=10, name="Fast")
        strategy2 = SimpleMAStrategy(fast_period=10, slow_period=20, name="Slow")
        
        combined = CombinedStrategy(
            strategies=[strategy1, strategy2],
            combination_method='majority'
        )
        
        combined_signal = combined.generate_signal(market_data, position)
        print(f"Combined signal: {combined_signal}")
        print(f"Metadata: {combined_signal.metadata}\n")
        
        # Test signal types
        print("3. Testing all signal types...")
        for sig_type in Signal:
            test_signal = SignalOutput(
                signal=sig_type,
                confidence=0.75,
                metadata={'test': True},
                strategy_name='TestStrategy'
            )
            print(f"  {test_signal}")
        
        print("\n✓ All examples completed successfully")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install pandas: pip install pandas numpy")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Exception in example")
"""
state.py - Trading Platform State Manager

Manages global runtime state for the trading platform including environment mode,
trading flags, and configuration state. Thread-safe and async-safe implementation.
"""

import asyncio
from enum import Enum
from typing import Any, Dict, Optional
from threading import Lock
from datetime import datetime


class Environment(Enum):
    """Trading environment modes."""
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


class TradingState:
    """
    Thread-safe and async-safe state manager for trading platform.
    
    Tracks runtime environment, global flags, and configuration state.
    Does not contain trading logic - purely for state management.
    
    Attributes:
        environment: Current trading environment (paper/testnet/live)
        is_trading_active: Global flag indicating if trading is enabled
        metadata: Additional key-value storage for runtime configuration
    """
    
    def __init__(self, environment: Environment = Environment.PAPER):
        """
        Initialize trading state manager.
        
        Args:
            environment: Initial environment mode (default: PAPER)
        """
        self._lock = Lock()
        self._environment = environment
        self._is_trading_active = False
        self._metadata: Dict[str, Any] = {}
        self._state_history: list = []
        self._created_at = datetime.utcnow()
        
    # Environment Management
    
    def get_environment(self) -> Environment:
        """
        Get current trading environment.
        
        Returns:
            Current Environment enum value
        """
        with self._lock:
            return self._environment
    
    def set_environment(self, environment: Environment) -> None:
        """
        Set trading environment.
        
        Args:
            environment: Target environment mode
            
        Raises:
            ValueError: If environment is not a valid Environment enum
        """
        if not isinstance(environment, Environment):
            raise ValueError(f"Invalid environment: {environment}. Must be Environment enum.")
        
        with self._lock:
            old_env = self._environment
            self._environment = environment
            self._log_state_change(f"Environment changed from {old_env.value} to {environment.value}")
    
    # Trading Active Flag Management
    
    def is_trading_active(self) -> bool:
        """
        Check if trading is currently active.
        
        Returns:
            True if trading is active, False otherwise
        """
        with self._lock:
            return self._is_trading_active
    
    def set_trading_active(self, active: bool) -> None:
        """
        Set trading active flag.
        
        Args:
            active: True to enable trading, False to disable
        """
        with self._lock:
            old_state = self._is_trading_active
            self._is_trading_active = active
            self._log_state_change(f"Trading active changed from {old_state} to {active}")
    
    def enable_trading(self) -> None:
        """Enable trading (convenience method)."""
        self.set_trading_active(True)
    
    def disable_trading(self) -> None:
        """Disable trading (convenience method)."""
        self.set_trading_active(False)
    
    # Metadata Management
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value by key.
        
        Args:
            key: Metadata key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Metadata value or default if key not found
        """
        with self._lock:
            return self._metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata key-value pair.
        
        Args:
            key: Metadata key
            value: Value to store
        """
        with self._lock:
            self._metadata[key] = value
            self._log_state_change(f"Metadata updated: {key}")
    
    def delete_metadata(self, key: str) -> bool:
        """
        Delete metadata key.
        
        Args:
            key: Metadata key to delete
            
        Returns:
            True if key existed and was deleted, False otherwise
        """
        with self._lock:
            if key in self._metadata:
                del self._metadata[key]
                self._log_state_change(f"Metadata deleted: {key}")
                return True
            return False
    
    def get_all_metadata(self) -> Dict[str, Any]:
        """
        Get copy of all metadata.
        
        Returns:
            Dictionary copy of all metadata
        """
        with self._lock:
            return self._metadata.copy()
    
    def clear_metadata(self) -> None:
        """Clear all metadata."""
        with self._lock:
            self._metadata.clear()
            self._log_state_change("All metadata cleared")
    
    # State Inspection
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get complete state snapshot.
        
        Returns:
            Dictionary containing all current state
        """
        with self._lock:
            return {
                "environment": self._environment.value,
                "is_trading_active": self._is_trading_active,
                "metadata": self._metadata.copy(),
                "created_at": self._created_at.isoformat(),
                "snapshot_time": datetime.utcnow().isoformat()
            }
    
    def get_state_history(self, limit: Optional[int] = None) -> list:
        """
        Get state change history.
        
        Args:
            limit: Maximum number of history entries to return (None for all)
            
        Returns:
            List of state change events
        """
        with self._lock:
            if limit is None:
                return self._state_history.copy()
            return self._state_history[-limit:]
    
    # Async-safe versions
    
    async def async_get_environment(self) -> Environment:
        """Async-safe environment getter."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get_environment
        )
    
    async def async_set_environment(self, environment: Environment) -> None:
        """Async-safe environment setter."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.set_environment, environment
        )
    
    async def async_is_trading_active(self) -> bool:
        """Async-safe trading active check."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.is_trading_active
        )
    
    async def async_set_trading_active(self, active: bool) -> None:
        """Async-safe trading active setter."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.set_trading_active, active
        )
    
    async def async_get_metadata(self, key: str, default: Any = None) -> Any:
        """Async-safe metadata getter."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get_metadata, key, default
        )
    
    async def async_set_metadata(self, key: str, value: Any) -> None:
        """Async-safe metadata setter."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.set_metadata, key, value
        )
    
    # Private helper methods
    
    def _log_state_change(self, message: str) -> None:
        """
        Log state change to history.
        
        Args:
            message: Description of state change
        """
        self._state_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message
        })
        # Keep history limited to last 1000 entries
        if len(self._state_history) > 1000:
            self._state_history = self._state_history[-1000:]
    
    def __repr__(self) -> str:
        """String representation of state."""
        with self._lock:
            return (f"TradingState(environment={self._environment.value}, "
                   f"is_trading_active={self._is_trading_active}, "
                   f"metadata_keys={list(self._metadata.keys())})")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    print("=== Trading State Manager Example ===\n")
    
    # Initialize state manager
    state = TradingState(environment=Environment.PAPER)
    print(f"Initial state: {state}")
    print(f"Environment: {state.get_environment().value}")
    print(f"Trading active: {state.is_trading_active()}\n")
    
    # Enable trading
    state.enable_trading()
    print(f"Trading enabled: {state.is_trading_active()}\n")
    
    # Change environment
    state.set_environment(Environment.TESTNET)
    print(f"Environment changed to: {state.get_environment().value}\n")
    
    # Set metadata
    state.set_metadata("max_position_size", 10000)
    state.set_metadata("risk_per_trade", 0.02)
    state.set_metadata("api_key_loaded", True)
    print(f"Metadata set: {state.get_all_metadata()}\n")
    
    # Get specific metadata
    max_pos = state.get_metadata("max_position_size")
    print(f"Max position size: {max_pos}\n")
    
    # Get state snapshot
    snapshot = state.get_state_snapshot()
    print("State snapshot:")
    for key, value in snapshot.items():
        print(f"  {key}: {value}")
    print()
    
    # View state history
    print("State change history:")
    for event in state.get_state_history(limit=5):
        print(f"  [{event['timestamp']}] {event['message']}")
    print()
    
    # Async example
    async def async_example():
        print("=== Async Operations Example ===\n")
        
        # Async get environment
        env = await state.async_get_environment()
        print(f"Async get environment: {env.value}")
        
        # Async set metadata
        await state.async_set_metadata("async_test", "success")
        value = await state.async_get_metadata("async_test")
        print(f"Async metadata set and retrieved: {value}")
        
        # Async trading control
        await state.async_set_trading_active(False)
        is_active = await state.async_is_trading_active()
        print(f"Trading active (async): {is_active}\n")
    
    # Run async example
    asyncio.run(async_example())
    
    # Final state
    print(f"Final state: {state}")
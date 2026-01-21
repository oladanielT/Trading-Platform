"""
broker.py - Trade Execution Engine

Executes approved trades in paper trading mode, handles order placement,
partial fills, retries, and execution confirmation. Does not contain
strategy logic - purely execution and order management.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

if TYPE_CHECKING:
    from strategies.base import PositionState


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionMode(Enum):
    """Execution mode."""
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


@dataclass
class OrderRequest:
    """
    Order request structure.
    
    Attributes:
        symbol: Trading pair symbol
        side: Buy or sell
        order_type: Type of order
        quantity: Amount to trade
        price: Limit price (optional)
        stop_price: Stop price (optional)
        time_in_force: Time in force (GTC, IOC, FOK)
        client_order_id: Client-generated order ID
        metadata: Additional order information
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate client order ID if not provided."""
        if self.client_order_id is None:
            self.client_order_id = f"order_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'client_order_id': self.client_order_id,
            'metadata': self.metadata
        }


@dataclass
class ExecutionResult:
    """
    Order execution result.
    
    Attributes:
        success: Whether execution was successful
        order_id: Exchange order ID
        client_order_id: Client order ID
        status: Current order status
        filled_quantity: Amount filled
        remaining_quantity: Amount remaining
        average_price: Average fill price
        total_cost: Total cost/proceeds
        fees: Trading fees
        fills: List of individual fills
        error_message: Error message if failed
        timestamp: Execution timestamp
        metadata: Additional execution details
    """
    success: bool
    order_id: str
    client_order_id: str
    status: OrderStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: float = 0.0
    total_cost: float = 0.0
    fees: float = 0.0
    fills: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    def is_partial(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_price': self.average_price,
            'total_cost': self.total_cost,
            'fees': self.fees,
            'fills': self.fills,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation."""
        if not self.success:
            return f"ExecutionResult(success=False, error={self.error_message})"
        return (f"ExecutionResult(status={self.status.value}, "
                f"filled={self.filled_quantity:.4f} @ ${self.average_price:.2f})")


class Broker:
    """
    Unified broker interface for trading platform.
    
    Provides position tracking, equity management, and order execution across
    different execution modes (paper/testnet/live). Acts as facade over
    PaperBroker with additional state tracking for positions and cash.
    """
    
    def __init__(
        self,
        exchange: Optional[Any] = None,
        mode: ExecutionMode = ExecutionMode.PAPER,
        initial_cash: float = 100000.0,
        slippage_bps: float = 5.0,
        maker_fee_bps: float = 10.0,
        taker_fee_bps: float = 10.0
    ):
        """
        Initialize broker.
        
        Args:
            exchange: ExchangeConnector instance (optional for paper trading)
            mode: Execution mode (paper/testnet/live)
            initial_cash: Starting cash balance
            slippage_bps: Slippage in basis points
            maker_fee_bps: Maker fee in basis points
            taker_fee_bps: Taker fee in basis points
        """
        self.exchange = exchange
        self.mode = mode
        self.initial_cash = initial_cash
        self.cash = initial_cash
        
        # Position tracking: symbol -> {'size': float, 'avg_price': float, 'side': str}
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize underlying paper broker
        self._paper_broker = PaperBroker(
            execution_mode=mode,
            slippage_bps=slippage_bps,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps
        )
        
        logger.info(
            f"Broker initialized: mode={mode.value}, "
            f"initial_cash=${initial_cash:,.2f}"
        )
    
    def get_position(self, symbol: str) -> "PositionState":
        """
        Get current position state for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            PositionState with current position information
        """
        from strategies.base import PositionState
        
        if symbol not in self.positions:
            return PositionState(has_position=False)
        
        pos = self.positions[symbol]
        return PositionState(
            has_position=True,
            position_type=pos['side'],  # 'long' or 'short'
            entry_price=pos['avg_price'],
            position_size=pos['size'],
            metadata={'symbol': symbol}
        )
    
    def get_equity(self) -> float:
        """
        Get current account equity (cash + position value).
        
        Returns:
            Total account equity
        """
        # For now, return cash (positions would need current prices to value)
        # In a full implementation, this would fetch current prices and calculate
        return self.cash
    
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Dict[str, Any]:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            quantity: Amount to trade
            
        Returns:
            Dictionary with execution results
        """
        # Get current price (simplified - would normally fetch from exchange)
        # For now, use a placeholder price
        current_price = 50000.0  # This should come from market data
        
        # Create order request
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        order = OrderRequest(
            symbol=symbol,
            side=order_side,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        
        # Execute via paper broker
        result = await self._paper_broker.execute_order(order, current_price)
        
        # Update position tracking if successful
        if result.success and result.status == OrderStatus.FILLED:
            self._update_position(
                symbol=symbol,
                side=side.lower(),
                quantity=result.filled_quantity,
                price=result.average_price
            )
            
            # Update cash
            if side.lower() == 'buy':
                self.cash -= result.total_cost + result.fees
            else:
                self.cash += result.total_cost - result.fees
        
        # Convert to dictionary format expected by main.py
        return {
            'status': 'filled' if result.status == OrderStatus.FILLED else result.status.value,
            'fill_price': result.average_price,
            'filled_quantity': result.filled_quantity,
            'order_id': result.order_id,
            'fees': result.fees,
            'success': result.success
        }
    
    def _update_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ):
        """
        Update position tracking.
        
        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Trade price
        """
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = {
                'size': quantity if side == 'buy' else -quantity,
                'avg_price': price,
                'side': 'long' if side == 'buy' else 'short'
            }
        else:
            # Update existing position
            pos = self.positions[symbol]
            old_size = pos['size']
            old_price = pos['avg_price']
            
            if side == 'buy':
                new_size = old_size + quantity
            else:
                new_size = old_size - quantity
            
            # Calculate new average price
            if abs(new_size) > 1e-8:
                if (old_size > 0 and side == 'buy') or (old_size < 0 and side == 'sell'):
                    # Adding to position - update average
                    total_cost_old = old_price * abs(old_size)
                    total_cost_new = price * quantity
                    pos['avg_price'] = (total_cost_old + total_cost_new) / abs(new_size)
                else:
                    # Reducing or reversing position
                    if abs(new_size) < abs(old_size):
                        # Reducing - keep old average
                        pass
                    else:
                        # Reversing - use new price
                        pos['avg_price'] = price
                
                pos['size'] = new_size
                pos['side'] = 'long' if new_size > 0 else 'short'
            else:
                # Position closed
                del self.positions[symbol]


class PaperBroker:
    """
    Paper trading broker for simulated order execution.
    
    Simulates realistic order execution including partial fills, slippage,
    and fees without connecting to real exchange. Useful for backtesting
    and strategy development.
    """
    
    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.PAPER,
        slippage_bps: float = 5.0,
        maker_fee_bps: float = 10.0,
        taker_fee_bps: float = 10.0,
        partial_fill_probability: float = 0.0,
        max_retry_attempts: int = 3,
        retry_delay_seconds: float = 1.0
    ):
        """
        Initialize paper broker.
        
        Args:
            execution_mode: Execution mode (paper/testnet/live)
            slippage_bps: Slippage in basis points (5 = 0.05%)
            maker_fee_bps: Maker fee in basis points
            taker_fee_bps: Taker fee in basis points
            partial_fill_probability: Probability of partial fill (0.0-1.0)
            max_retry_attempts: Maximum retry attempts for failed orders
            retry_delay_seconds: Delay between retries
        """
        self.execution_mode = execution_mode
        self.slippage_bps = slippage_bps
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.partial_fill_probability = partial_fill_probability
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        
        # Order tracking
        self.orders: Dict[str, ExecutionResult] = {}
        self.order_counter = 0
        
        logger.info(
            f"PaperBroker initialized: mode={execution_mode.value}, "
            f"slippage={slippage_bps}bps, fees={taker_fee_bps}bps"
        )
    
    async def execute_order(
        self,
        order_request: OrderRequest,
        current_price: float
    ) -> ExecutionResult:
        """
        Execute order with retry logic.
        
        Args:
            order_request: Order request details
            current_price: Current market price
            
        Returns:
            ExecutionResult with execution details
        """
        logger.info(
            f"Executing {order_request.side.value} order: "
            f"{order_request.quantity:.4f} {order_request.symbol} "
            f"@ ${current_price:.2f}"
        )
        
        # Validate order
        validation_error = self._validate_order(order_request, current_price)
        if validation_error:
            return self._create_rejected_result(
                order_request,
                validation_error
            )
        
        # Attempt execution with retries
        last_error = None
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                result = await self._execute_single_order(
                    order_request,
                    current_price
                )
                
                # Store order
                self.orders[result.order_id] = result
                
                if result.success:
                    logger.info(
                        f"Order executed successfully: {result.order_id} "
                        f"({result.status.value})"
                    )
                    return result
                else:
                    last_error = result.error_message
                    if attempt < self.max_retry_attempts:
                        logger.warning(
                            f"Order execution failed (attempt {attempt}): "
                            f"{last_error}. Retrying..."
                        )
                        await asyncio.sleep(self.retry_delay_seconds)
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Order execution error (attempt {attempt}): {e}")
                if attempt < self.max_retry_attempts:
                    await asyncio.sleep(self.retry_delay_seconds)
        
        # All retries exhausted
        logger.error(
            f"Order execution failed after {self.max_retry_attempts} attempts: "
            f"{last_error}"
        )
        return self._create_rejected_result(
            order_request,
            f"Failed after {self.max_retry_attempts} attempts: {last_error}"
        )
    
    async def _execute_single_order(
        self,
        order_request: OrderRequest,
        current_price: float
    ) -> ExecutionResult:
        """Execute single order attempt."""
        # Generate order ID
        self.order_counter += 1
        order_id = f"paper_{self.order_counter:06d}"
        
        # Calculate execution price with slippage
        execution_price = self._calculate_execution_price(
            current_price,
            order_request.side,
            order_request.order_type
        )
        
        # Simulate partial fill
        import random
        if random.random() < self.partial_fill_probability:
            fill_percentage = random.uniform(0.5, 0.95)
            filled_quantity = order_request.quantity * fill_percentage
            status = OrderStatus.PARTIAL
        else:
            filled_quantity = order_request.quantity
            status = OrderStatus.FILLED
        
        remaining_quantity = order_request.quantity - filled_quantity
        
        # Calculate costs and fees
        total_cost = filled_quantity * execution_price
        
        # Use taker fee for market orders, maker fee for limit orders
        fee_bps = (self.taker_fee_bps if order_request.order_type == OrderType.MARKET
                   else self.maker_fee_bps)
        fees = total_cost * (fee_bps / 10000)
        
        # Create fill record
        fill = {
            'price': execution_price,
            'quantity': filled_quantity,
            'fee': fees,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Create result
        result = ExecutionResult(
            success=True,
            order_id=order_id,
            client_order_id=order_request.client_order_id,
            status=status,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            average_price=execution_price,
            total_cost=total_cost,
            fees=fees,
            fills=[fill],
            metadata={
                'execution_mode': self.execution_mode.value,
                'original_price': current_price,
                'slippage_bps': self.slippage_bps,
                'symbol': order_request.symbol,
                'side': order_request.side.value,
                'order_type': order_request.order_type.value
            }
        )
        
        return result
    
    def _validate_order(
        self,
        order_request: OrderRequest,
        current_price: float
    ) -> Optional[str]:
        """
        Validate order request.
        
        Returns:
            Error message if invalid, None if valid
        """
        if order_request.quantity <= 0:
            return f"Invalid quantity: {order_request.quantity}"
        
        if current_price <= 0:
            return f"Invalid current price: {current_price}"
        
        # Validate limit price
        if order_request.order_type == OrderType.LIMIT:
            if order_request.price is None:
                return "Limit price required for limit order"
            if order_request.price <= 0:
                return f"Invalid limit price: {order_request.price}"
        
        # Validate stop price
        if order_request.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
            if order_request.stop_price is None:
                return "Stop price required for stop order"
            if order_request.stop_price <= 0:
                return f"Invalid stop price: {order_request.stop_price}"
        
        return None
    
    def _calculate_execution_price(
        self,
        current_price: float,
        side: OrderSide,
        order_type: OrderType
    ) -> float:
        """Calculate execution price with slippage."""
        if order_type != OrderType.MARKET:
            # Limit orders execute at limit price (in paper trading)
            return current_price
        
        # Apply slippage
        slippage_factor = self.slippage_bps / 10000
        
        if side == OrderSide.BUY:
            # Buying: price goes up (unfavorable)
            execution_price = current_price * (1 + slippage_factor)
        else:
            # Selling: price goes down (unfavorable)
            execution_price = current_price * (1 - slippage_factor)
        
        return execution_price
    
    def _create_rejected_result(
        self,
        order_request: OrderRequest,
        error_message: str
    ) -> ExecutionResult:
        """Create rejected order result."""
        self.order_counter += 1
        order_id = f"paper_rejected_{self.order_counter:06d}"
        
        return ExecutionResult(
            success=False,
            order_id=order_id,
            client_order_id=order_request.client_order_id,
            status=OrderStatus.REJECTED,
            error_message=error_message,
            metadata={
                'symbol': order_request.symbol,
                'side': order_request.side.value,
                'quantity': order_request.quantity
            }
        )
    
    def get_order(self, order_id: str) -> Optional[ExecutionResult]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            ExecutionResult or None if not found
        """
        return self.orders.get(order_id)
    
    def get_all_orders(self) -> List[ExecutionResult]:
        """
        Get all orders.
        
        Returns:
            List of all ExecutionResults
        """
        return list(self.orders.values())
    
    def get_filled_orders(self) -> List[ExecutionResult]:
        """
        Get all filled orders.
        
        Returns:
            List of filled ExecutionResults
        """
        return [
            order for order in self.orders.values()
            if order.status == OrderStatus.FILLED
        ]
    
    def get_order_stats(self) -> Dict[str, Any]:
        """
        Get order statistics.
        
        Returns:
            Dictionary with order statistics
        """
        total = len(self.orders)
        filled = sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED)
        partial = sum(1 for o in self.orders.values() if o.status == OrderStatus.PARTIAL)
        rejected = sum(1 for o in self.orders.values() if o.status == OrderStatus.REJECTED)
        
        total_volume = sum(
            o.total_cost for o in self.orders.values()
            if o.success
        )
        total_fees = sum(
            o.fees for o in self.orders.values()
            if o.success
        )
        
        return {
            'total_orders': total,
            'filled_orders': filled,
            'partial_orders': partial,
            'rejected_orders': rejected,
            'fill_rate': (filled / total * 100) if total > 0 else 0,
            'total_volume': total_volume,
            'total_fees': total_fees
        }


# Convenience functions

async def execute_trade(
    symbol: str,
    side: str,
    quantity: float,
    current_price: float,
    broker: Optional[PaperBroker] = None,
    order_type: str = "market"
) -> ExecutionResult:
    """
    Convenience function to execute a trade.
    
    Args:
        symbol: Trading pair symbol
        side: 'buy' or 'sell'
        quantity: Amount to trade
        current_price: Current market price
        broker: Broker instance (creates default if None)
        order_type: Order type (default: 'market')
        
    Returns:
        ExecutionResult
    """
    if broker is None:
        broker = PaperBroker()
    
    order = OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
        order_type=OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT,
        quantity=quantity
    )
    
    return await broker.execute_order(order, current_price)


# Example usage
if __name__ == "__main__":
    
    async def main():
        print("=== Paper Broker Example ===\n")
        
        # Initialize broker
        broker = PaperBroker(
            execution_mode=ExecutionMode.PAPER,
            slippage_bps=5.0,
            maker_fee_bps=10.0,
            taker_fee_bps=10.0,
            partial_fill_probability=0.2,
            max_retry_attempts=3
        )
        
        print(f"Broker initialized: {broker.execution_mode.value}\n")
        
        # Test 1: Market buy order
        print("1. Market Buy Order")
        print("-" * 50)
        
        buy_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.5
        )
        
        current_price = 50000.0
        result = await broker.execute_order(buy_order, current_price)
        
        print(f"Result: {result}")
        print(f"Success: {result.success}")
        print(f"Order ID: {result.order_id}")
        print(f"Status: {result.status.value}")
        print(f"Filled: {result.filled_quantity:.4f} BTC")
        print(f"Avg Price: ${result.average_price:.2f}")
        print(f"Total Cost: ${result.total_cost:.2f}")
        print(f"Fees: ${result.fees:.2f}")
        print(f"Slippage: ${result.average_price - current_price:.2f}\n")
        
        # Test 2: Market sell order
        print("2. Market Sell Order")
        print("-" * 50)
        
        sell_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.3
        )
        
        result_sell = await broker.execute_order(sell_order, current_price)
        print(f"Result: {result_sell}")
        print(f"Filled: {result_sell.filled_quantity:.4f} BTC")
        print(f"Avg Price: ${result_sell.average_price:.2f}")
        print(f"Total Proceeds: ${result_sell.total_cost:.2f}\n")
        
        # Test 3: Limit order
        print("3. Limit Buy Order")
        print("-" * 50)
        
        limit_order = OrderRequest(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=5.0,
            price=3000.0
        )
        
        result_limit = await broker.execute_order(limit_order, 3050.0)
        print(f"Result: {result_limit}")
        print(f"Executed at limit price: ${result_limit.average_price:.2f}\n")
        
        # Test 4: Invalid order (should be rejected)
        print("4. Invalid Order (Negative Quantity)")
        print("-" * 50)
        
        invalid_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=-1.0
        )
        
        result_invalid = await broker.execute_order(invalid_order, current_price)
        print(f"Result: {result_invalid}")
        print(f"Success: {result_invalid.success}")
        print(f"Status: {result_invalid.status.value}")
        print(f"Error: {result_invalid.error_message}\n")
        
        # Test 5: Multiple orders and statistics
        print("5. Executing Multiple Orders")
        print("-" * 50)
        
        for i in range(3):
            order = OrderRequest(
                symbol="BTC/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.1
            )
            await broker.execute_order(order, current_price)
        
        stats = broker.get_order_stats()
        print(f"Order Statistics:")
        print(f"  Total Orders: {stats['total_orders']}")
        print(f"  Filled Orders: {stats['filled_orders']}")
        print(f"  Partial Orders: {stats['partial_orders']}")
        print(f"  Rejected Orders: {stats['rejected_orders']}")
        print(f"  Fill Rate: {stats['fill_rate']:.1f}%")
        print(f"  Total Volume: ${stats['total_volume']:.2f}")
        print(f"  Total Fees: ${stats['total_fees']:.2f}\n")
        
        # Test 6: Retrieve specific order
        print("6. Retrieve Specific Order")
        print("-" * 50)
        
        retrieved = broker.get_order(result.order_id)
        if retrieved:
            print(f"Retrieved order: {result.order_id}")
            print(f"Status: {retrieved.status.value}")
            print(f"Filled: {retrieved.filled_quantity:.4f}\n")
        
        # Test 7: Get all filled orders
        print("7. All Filled Orders")
        print("-" * 50)
        
        filled_orders = broker.get_filled_orders()
        print(f"Total filled orders: {len(filled_orders)}")
        for order in filled_orders[:3]:
            print(f"  {order.order_id}: {order.metadata.get('side')} "
                  f"{order.filled_quantity:.4f} @ ${order.average_price:.2f}")
        
        print("\n✓ All examples completed successfully")
        print("\nNote: This is paper trading - no real trades executed.")
    
    # Run async example
    asyncio.run(main())
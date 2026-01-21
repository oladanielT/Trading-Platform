"""
order_manager.py - Order Queue and Management System

Manages order queue, tracks active orders, handles retry logic for failed
executions, and provides comprehensive logging. Does not make strategy
decisions - purely operational order management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import uuid


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderPriority(Enum):
    """Order priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class QueueStatus(Enum):
    """Order queue status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class QueuedOrder:
    """
    Order in management queue.
    
    Attributes:
        order_id: Unique order identifier
        order_request: Order request details
        current_price: Price at time of queuing
        priority: Order priority level
        status: Current queue status
        attempts: Number of execution attempts
        max_attempts: Maximum retry attempts
        created_at: Time order was queued
        last_attempt: Time of last execution attempt
        next_retry: Time of next retry attempt
        error_history: List of error messages from failed attempts
        execution_result: Final execution result if completed
        metadata: Additional order context
    """
    order_id: str
    order_request: Dict[str, Any]
    current_price: float
    priority: OrderPriority = OrderPriority.NORMAL
    status: QueueStatus = QueueStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    error_history: List[str] = field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_retry(self) -> bool:
        """Check if order can be retried."""
        return (self.attempts < self.max_attempts and 
                self.status in [QueueStatus.FAILED, QueueStatus.RETRYING])
    
    def is_ready_for_retry(self) -> bool:
        """Check if order is ready for retry attempt."""
        if not self.can_retry():
            return False
        if self.next_retry is None:
            return True
        return datetime.utcnow() >= self.next_retry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'order_request': self.order_request,
            'current_price': self.current_price,
            'priority': self.priority.name,
            'status': self.status.value,
            'attempts': self.attempts,
            'max_attempts': self.max_attempts,
            'created_at': self.created_at.isoformat(),
            'last_attempt': self.last_attempt.isoformat() if self.last_attempt else None,
            'next_retry': self.next_retry.isoformat() if self.next_retry else None,
            'error_history': self.error_history,
            'execution_result': self.execution_result,
            'metadata': self.metadata
        }
    
    def __lt__(self, other):
        """Compare for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at


@dataclass
class OrderManagerStats:
    """Order manager statistics."""
    total_orders: int = 0
    pending_orders: int = 0
    processing_orders: int = 0
    completed_orders: int = 0
    failed_orders: int = 0
    cancelled_orders: int = 0
    retrying_orders: int = 0
    total_attempts: int = 0
    success_rate: float = 0.0
    average_attempts: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_orders': self.total_orders,
            'pending_orders': self.pending_orders,
            'processing_orders': self.processing_orders,
            'completed_orders': self.completed_orders,
            'failed_orders': self.failed_orders,
            'cancelled_orders': self.cancelled_orders,
            'retrying_orders': self.retrying_orders,
            'total_attempts': self.total_attempts,
            'success_rate': self.success_rate,
            'average_attempts': self.average_attempts
        }


class OrderManager:
    """
    Order queue and management system.
    
    Manages order lifecycle from queuing through execution, handles retries
    for failed orders, and provides comprehensive tracking and logging.
    Does not make trading decisions - purely operational management.
    """
    
    def __init__(
        self,
        max_concurrent_orders: int = 5,
        default_max_attempts: int = 3,
        retry_delay_seconds: float = 5.0,
        exponential_backoff: bool = True,
        max_queue_size: Optional[int] = None
    ):
        """
        Initialize order manager.
        
        Args:
            max_concurrent_orders: Maximum concurrent order executions
            default_max_attempts: Default maximum retry attempts per order
            retry_delay_seconds: Base delay between retry attempts
            exponential_backoff: Use exponential backoff for retries
            max_queue_size: Maximum queue size (None = unlimited)
        """
        self.max_concurrent_orders = max_concurrent_orders
        self.default_max_attempts = default_max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.exponential_backoff = exponential_backoff
        self.max_queue_size = max_queue_size
        
        # Order storage
        self.orders: Dict[str, QueuedOrder] = {}
        self.pending_queue: deque = deque()
        self.processing_orders: Dict[str, QueuedOrder] = {}
        
        # Processing state
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.total_orders_processed = 0
        self.total_execution_attempts = 0
        
        logger.info(
            f"OrderManager initialized: max_concurrent={max_concurrent_orders}, "
            f"max_attempts={default_max_attempts}"
        )
    
    def queue_order(
        self,
        order_request: Dict[str, Any],
        current_price: float,
        priority: OrderPriority = OrderPriority.NORMAL,
        max_attempts: Optional[int] = None,
        **metadata
    ) -> str:
        """
        Add order to queue.
        
        Args:
            order_request: Order request details
            current_price: Current market price
            priority: Order priority level
            max_attempts: Maximum retry attempts (uses default if None)
            **metadata: Additional metadata
            
        Returns:
            Order ID
            
        Raises:
            ValueError: If queue is full
        """
        # Check queue size limit
        if self.max_queue_size and len(self.pending_queue) >= self.max_queue_size:
            raise ValueError(
                f"Queue full: {len(self.pending_queue)}/{self.max_queue_size}"
            )
        
        # Create queued order
        order_id = f"om_{uuid.uuid4().hex[:12]}"
        
        queued_order = QueuedOrder(
            order_id=order_id,
            order_request=order_request,
            current_price=current_price,
            priority=priority,
            max_attempts=max_attempts or self.default_max_attempts,
            metadata=metadata
        )
        
        # Add to storage and queue
        self.orders[order_id] = queued_order
        self.pending_queue.append(queued_order)
        
        logger.info(
            f"Order queued: {order_id} ({priority.name} priority) - "
            f"{order_request.get('side', 'UNKNOWN')} {order_request.get('symbol', 'UNKNOWN')}"
        )
        
        return order_id
    
    async def start_processing(
        self,
        executor_func: Callable
    ) -> None:
        """
        Start processing orders from queue.
        
        Args:
            executor_func: Async function to execute orders
                          Should accept (order_request, current_price) and return execution result
        """
        if self.is_running:
            logger.warning("Order processing already running")
            return
        
        self.is_running = True
        logger.info("Starting order processing")
        
        # Create processing tasks
        for i in range(self.max_concurrent_orders):
            task = asyncio.create_task(
                self._process_orders_worker(executor_func, worker_id=i+1)
            )
            self.processing_tasks.append(task)
        
        logger.info(f"Started {len(self.processing_tasks)} processing workers")
    
    async def stop_processing(self) -> None:
        """Stop processing orders."""
        if not self.is_running:
            return
        
        logger.info("Stopping order processing")
        self.is_running = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        self.processing_tasks.clear()
        logger.info("Order processing stopped")
    
    async def _process_orders_worker(
        self,
        executor_func: Callable,
        worker_id: int
    ) -> None:
        """Worker coroutine to process orders from queue."""
        logger.debug(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get next order from queue
                order = await self._get_next_order()
                
                if order is None:
                    # Queue empty, wait before checking again
                    await asyncio.sleep(0.5)
                    continue
                
                # Process order
                await self._process_order(order, executor_func, worker_id)
                
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1.0)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _get_next_order(self) -> Optional[QueuedOrder]:
        """Get next order from queue with priority."""
        # Check for orders ready for retry first
        for order in list(self.orders.values()):
            if order.is_ready_for_retry():
                order.status = QueueStatus.PENDING
                return order
        
        # Get from pending queue
        if self.pending_queue:
            # Sort by priority (handled by QueuedOrder.__lt__)
            pending_list = list(self.pending_queue)
            pending_list.sort()
            
            # Get highest priority order
            order = pending_list[0]
            self.pending_queue.remove(order)
            
            return order
        
        return None
    
    async def _process_order(
        self,
        order: QueuedOrder,
        executor_func: Callable,
        worker_id: int
    ) -> None:
        """Process single order execution."""
        order.status = QueueStatus.PROCESSING
        order.attempts += 1
        order.last_attempt = datetime.utcnow()
        self.total_execution_attempts += 1
        
        # Add to processing dict
        self.processing_orders[order.order_id] = order
        
        logger.info(
            f"[Worker {worker_id}] Processing order {order.order_id} "
            f"(attempt {order.attempts}/{order.max_attempts})"
        )
        
        try:
            # Execute order
            result = await executor_func(
                order.order_request,
                order.current_price
            )
            
            # Check execution result
            if result.get('success', False):
                # Success
                order.status = QueueStatus.COMPLETED
                order.execution_result = result
                self.total_orders_processed += 1
                
                logger.info(
                    f"[Worker {worker_id}] Order {order.order_id} completed successfully"
                )
            else:
                # Failed
                error_msg = result.get('error_message', 'Unknown error')
                order.error_history.append(error_msg)
                
                # Check if can retry
                if order.can_retry():
                    order.status = QueueStatus.RETRYING
                    
                    # Calculate retry delay
                    if self.exponential_backoff:
                        delay = self.retry_delay_seconds * (2 ** (order.attempts - 1))
                    else:
                        delay = self.retry_delay_seconds
                    
                    order.next_retry = datetime.utcnow() + timedelta(seconds=delay)
                    
                    logger.warning(
                        f"[Worker {worker_id}] Order {order.order_id} failed "
                        f"(attempt {order.attempts}): {error_msg}. "
                        f"Retrying in {delay:.1f}s"
                    )
                else:
                    # Max attempts reached
                    order.status = QueueStatus.FAILED
                    order.execution_result = result
                    
                    logger.error(
                        f"[Worker {worker_id}] Order {order.order_id} failed "
                        f"after {order.attempts} attempts: {error_msg}"
                    )
        
        except Exception as e:
            # Execution exception
            error_msg = str(e)
            order.error_history.append(error_msg)
            
            if order.can_retry():
                order.status = QueueStatus.RETRYING
                delay = self.retry_delay_seconds
                order.next_retry = datetime.utcnow() + timedelta(seconds=delay)
                
                logger.error(
                    f"[Worker {worker_id}] Order {order.order_id} exception: {e}. "
                    f"Retrying in {delay:.1f}s",
                    exc_info=True
                )
            else:
                order.status = QueueStatus.FAILED
                logger.error(
                    f"[Worker {worker_id}] Order {order.order_id} failed "
                    f"with exception after {order.attempts} attempts: {e}",
                    exc_info=True
                )
        
        finally:
            # Remove from processing dict
            self.processing_orders.pop(order.order_id, None)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending or retrying order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled, False if not found or already processing/completed
        """
        order = self.orders.get(order_id)
        
        if not order:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        if order.status in [QueueStatus.COMPLETED, QueueStatus.FAILED]:
            logger.warning(f"Cannot cancel order in {order.status.value} state: {order_id}")
            return False
        
        if order.status == QueueStatus.PROCESSING:
            logger.warning(f"Cannot cancel order currently processing: {order_id}")
            return False
        
        # Remove from queue if present
        try:
            self.pending_queue.remove(order)
        except ValueError:
            pass
        
        order.status = QueueStatus.CANCELLED
        logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    def get_order(self, order_id: str) -> Optional[QueuedOrder]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            QueuedOrder or None if not found
        """
        return self.orders.get(order_id)
    
    def get_orders_by_status(self, status: QueueStatus) -> List[QueuedOrder]:
        """
        Get all orders with specific status.
        
        Args:
            status: Queue status to filter by
            
        Returns:
            List of matching orders
        """
        return [
            order for order in self.orders.values()
            if order.status == status
        ]
    
    def get_stats(self) -> OrderManagerStats:
        """
        Get current statistics.
        
        Returns:
            OrderManagerStats with current metrics
        """
        orders_list = list(self.orders.values())
        total = len(orders_list)
        
        pending = sum(1 for o in orders_list if o.status == QueueStatus.PENDING)
        processing = sum(1 for o in orders_list if o.status == QueueStatus.PROCESSING)
        completed = sum(1 for o in orders_list if o.status == QueueStatus.COMPLETED)
        failed = sum(1 for o in orders_list if o.status == QueueStatus.FAILED)
        cancelled = sum(1 for o in orders_list if o.status == QueueStatus.CANCELLED)
        retrying = sum(1 for o in orders_list if o.status == QueueStatus.RETRYING)
        
        success_rate = (completed / total * 100) if total > 0 else 0.0
        avg_attempts = (self.total_execution_attempts / total) if total > 0 else 0.0
        
        return OrderManagerStats(
            total_orders=total,
            pending_orders=pending,
            processing_orders=processing,
            completed_orders=completed,
            failed_orders=failed,
            cancelled_orders=cancelled,
            retrying_orders=retrying,
            total_attempts=self.total_execution_attempts,
            success_rate=success_rate,
            average_attempts=avg_attempts
        )
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get detailed queue status.
        
        Returns:
            Dictionary with queue metrics
        """
        stats = self.get_stats()
        
        return {
            'is_running': self.is_running,
            'pending_queue_size': len(self.pending_queue),
            'processing_count': len(self.processing_orders),
            'max_concurrent': self.max_concurrent_orders,
            'stats': stats.to_dict()
        }
    
    def clear_completed_orders(self) -> int:
        """
        Clear completed and failed orders from storage.
        
        Returns:
            Number of orders cleared
        """
        to_remove = [
            order_id for order_id, order in self.orders.items()
            if order.status in [QueueStatus.COMPLETED, QueueStatus.FAILED, QueueStatus.CANCELLED]
        ]
        
        for order_id in to_remove:
            del self.orders[order_id]
        
        logger.info(f"Cleared {len(to_remove)} completed/failed orders")
        return len(to_remove)


# Example usage
if __name__ == "__main__":
    
    async def mock_executor(order_request: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Mock order executor for testing."""
        import random
        
        # Simulate execution delay
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Simulate 80% success rate
        if random.random() < 0.8:
            return {
                'success': True,
                'order_id': f"exec_{uuid.uuid4().hex[:8]}",
                'filled_quantity': order_request.get('quantity', 0),
                'average_price': current_price
            }
        else:
            return {
                'success': False,
                'error_message': 'Simulated execution failure'
            }
    
    async def main():
        print("=== Order Manager Example ===\n")
        
        # Initialize manager
        manager = OrderManager(
            max_concurrent_orders=3,
            default_max_attempts=3,
            retry_delay_seconds=2.0,
            exponential_backoff=True
        )
        
        print(f"Order manager initialized\n")
        
        # Queue some orders
        print("1. Queuing Orders")
        print("-" * 50)
        
        order_ids = []
        for i in range(5):
            priority = OrderPriority.HIGH if i == 0 else OrderPriority.NORMAL
            
            order_id = manager.queue_order(
                order_request={
                    'symbol': 'BTC/USDT',
                    'side': 'BUY' if i % 2 == 0 else 'SELL',
                    'quantity': 0.1,
                    'type': 'market'
                },
                current_price=50000.0,
                priority=priority,
                trader_id=f"trader_{i}"
            )
            
            order_ids.append(order_id)
            print(f"Queued order {i+1}: {order_id} ({priority.name})")
        
        print()
        
        # Start processing
        print("2. Starting Order Processing")
        print("-" * 50)
        
        await manager.start_processing(mock_executor)
        
        # Monitor for a bit
        print("Monitoring order processing...\n")
        
        for _ in range(10):
            await asyncio.sleep(1)
            status = manager.get_queue_status()
            stats = status['stats']
            
            print(f"Queue: {status['pending_queue_size']} pending, "
                  f"{status['processing_count']} processing | "
                  f"Completed: {stats['completed_orders']}, "
                  f"Failed: {stats['failed_orders']}, "
                  f"Retrying: {stats['retrying_orders']}")
            
            # Stop if all processed
            if (stats['completed_orders'] + stats['failed_orders'] == len(order_ids)):
                break
        
        print()
        
        # Get final statistics
        print("3. Final Statistics")
        print("-" * 50)
        
        stats = manager.get_stats()
        print(f"Total Orders: {stats.total_orders}")
        print(f"Completed: {stats.completed_orders}")
        print(f"Failed: {stats.failed_orders}")
        print(f"Success Rate: {stats.success_rate:.1f}%")
        print(f"Total Attempts: {stats.total_attempts}")
        print(f"Average Attempts: {stats.average_attempts:.2f}\n")
        
        # Check individual orders
        print("4. Individual Order Status")
        print("-" * 50)
        
        for order_id in order_ids[:3]:
            order = manager.get_order(order_id)
            if order:
                print(f"{order_id}:")
                print(f"  Status: {order.status.value}")
                print(f"  Attempts: {order.attempts}")
                if order.error_history:
                    print(f"  Errors: {order.error_history}")
        
        print()
        
        # Stop processing
        print("5. Stopping Order Processing")
        print("-" * 50)
        
        await manager.stop_processing()
        print("Processing stopped\n")
        
        # Clean up
        cleared = manager.clear_completed_orders()
        print(f"6. Cleaned up {cleared} completed orders\n")
        
        print("✓ All examples completed successfully")
        print("\nNote: This module manages order execution - it does not make trading decisions.")
    
    # Run async example
    asyncio.run(main())
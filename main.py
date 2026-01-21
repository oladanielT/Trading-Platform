"""
main.py - Trading Platform Entry Point

Orchestrates the modular trading platform following strict separation of concerns.
This file contains ONLY orchestration logic - no strategy, risk, or execution logic.

Module Loading Order:
1. Core (config & environment)
2. Monitoring (logging & metrics)
3. Data (market feed)
4. Strategy (signal generation)
5. Risk (position sizing & drawdown guard)
6. Execution (broker & order management)

Supports: paper/testnet/live modes with graceful shutdown on SIGINT/SIGTERM
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None

# Core imports
from core.environment import EnvironmentManager
from core.state import TradingState, Environment

# Monitoring imports
from monitoring.logger import LoggerManager
from monitoring.metrics import MetricsCollector

# Data imports
from data.exchange import ExchangeConnector
from data.market_feed import MarketFeed

# Strategy imports
from strategies.base import BaseStrategy, Signal
from strategies.ema_trend import EMATrendStrategy
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.regime_detector import MarketRegime
from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.pattern_strategies import (
    ContinuationStrategy,
    FVGStrategy,
    ABCStrategy,
    ReclamationBlockStrategy,
)
from strategies.range_trading import RangeTradingStrategy

# Risk imports
from risk.position_sizing import PositionSizer, SizingMethod
from risk.drawdown_guard import DrawdownGuard
from risk.portfolio_risk_manager import PortfolioRiskManager
from risk.regime_capital_allocator import RegimeCapitalAllocator

# Execution imports
from execution.broker import Broker, ExecutionMode
from execution.order_manager import OrderManager
from execution.execution_reality_engine import ExecutionRealityEngine
from execution.sim_live_executor import SimLiveExecutor

# Optimization imports (AI-driven performance enhancement)
from ai.optimization_engine import initialize_engine, get_engine
from ai.performance_decomposer import record_trade
from ai.optimization_monitor import get_monitor


# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingPlatform:
    """
    Main orchestrator for the trading platform.
    
    Coordinates module initialization, trading loop execution, and graceful shutdown.
    Contains no trading logic - purely orchestration.
    """
    
    def __init__(self, config_path: str = "core/config.yaml", overrides: Optional[Dict[str, Any]] = None, run_once: bool = False):

        # Path to configuration YAML file
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        # CLI / programmatic overrides applied after config load
        self.cli_overrides: Dict[str, Any] = overrides or {}
        # If set, perform only a single trading loop iteration
        self.run_once = run_once
        self.shutdown_requested = False
        
        # Module instances (initialized in setup)
        self.state: Optional[TradingState] = None
        self.env_manager: Optional[EnvironmentManager] = None
        self.logger_mgr: Optional[LoggerManager] = None
        self.metrics: Optional[MetricsCollector] = None
        self.exchange: Optional[ExchangeConnector] = None
        self.market_feed: Optional[MarketFeed] = None
        self.strategy: Optional[BaseStrategy] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.drawdown_guard: Optional[DrawdownGuard] = None
        self.portfolio_risk_manager: Optional[PortfolioRiskManager] = None
        self.regime_capital_allocator: Optional[RegimeCapitalAllocator] = None
        self.broker: Optional[Broker] = None
        self.order_manager: Optional[OrderManager] = None
        self.execution_reality_engine: Optional[ExecutionRealityEngine] = None
        
        # Optimization module (AI-driven performance enhancement)
        self.optimization_engine = None
        self.optimization_monitor = None
        self.optimization_enabled = False
        
        # Trade tracking for optimization (entry prices and metadata)
        self._open_trades: Dict[str, Dict[str, Any]] = {}
        
    def load_config(self) -> Dict[str, Any]:
        # Load configuration from YAML file.
        # Returns: Configuration dictionary
        # Raises: FileNotFoundError if config file not found; ValueError if YAML parsing fails
        logger.info(f"Loading configuration from {self.config_path}")
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
        
        if yaml is None:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info("Configuration loaded successfully")
            return {**self._default_config(), **config}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise ValueError(f"Invalid configuration file: {e}")
    
    def _default_config(self) -> Dict[str, Any]:
        # Return default configuration.
        # Returns: Default configuration dictionary
        return {
            'environment': 'paper',
            'exchange': {
                'id': 'binance',
                'testnet': True,
            },
            'trading': {
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'initial_capital': 100000.0,
                'fetch_limit': 500,
            },
            'strategy': {
                'name': 'ema_trend',
                'fast_period': 50,
                'slow_period': 200,
                'signal_confidence': 0.7,
            },
            'risk': {
                'max_position_size': 0.1,  # 10% of equity
                'risk_per_trade': 0.02,    # 2% risk per trade
                'max_daily_drawdown': 0.05,  # 5% max daily loss
                'max_total_drawdown': 0.20,  # 20% max total loss
                'sizing_method': 'fixed_risk',
            },
            'execution': {
                'order_timeout': 30,
                'max_retries': 3,
                'retry_delay': 5,
            },
            'monitoring': {
                'log_level': 'INFO',
                'log_file': 'logs/trading.log',
            },
            'optimization': {
                'enabled': False,  # Set to True to enable optimization
                'enable_quality_filtering': True,
                'enable_adaptive_weighting': True,
                'enable_retirement_management': True,
                'update_interval_seconds': 3600,  # 1 hour
            },
            'portfolio_risk': {
                'enabled': False,  # Set to True to enable portfolio-level risk management
                'correlation_lookback': 100,
                'max_pairwise_correlation': 0.85,
                'max_symbol_exposure_pct': 0.25,
                'max_directional_exposure_pct': 0.60,
                'max_regime_exposure_pct': 0.50,
            }
        }
    
    async def setup(self):
        """
        Initialize all modules in correct order.
        
        Module Order: Core → Monitoring → Data → Strategy → Risk → Execution
        """
        logger.info("=" * 80)
        logger.info("TRADING PLATFORM INITIALIZATION")
        logger.info("=" * 80)
        
        # Load configuration
        self.config = self.load_config()

        # Apply CLI / programmatic overrides (if any)
        if self.cli_overrides:
            env_override = self.cli_overrides.get('environment')
            if env_override:
                self.config['environment'] = env_override

            if 'optimization_enabled' in self.cli_overrides:
                opt_cfg = self.config.get('optimization', {})
                opt_cfg['enabled'] = bool(self.cli_overrides['optimization_enabled'])
                self.config['optimization'] = opt_cfg

            log_level = self.cli_overrides.get('log_level')
            if log_level:
                mon = self.config.get('monitoring', {})
                mon['log_level'] = log_level
                self.config['monitoring'] = mon

            exchange_id = self.cli_overrides.get('exchange_id')
            if exchange_id:
                exch = self.config.get('exchange', {})
                exch['id'] = exchange_id
                self.config['exchange'] = exch
        
        # Stage 1: Core Module
        logger.info("[1/6] Initializing Core Module (Environment & State)")
        await self._setup_core()
        
        # Stage 2: Monitoring Module
        logger.info("[2/6] Initializing Monitoring Module (Logger & Metrics)")
        await self._setup_monitoring()
        
        # Stage 3: Data Module
        logger.info("[3/6] Initializing Data Module (Exchange & Market Feed)")
        await self._setup_data()
        
        # Stage 4: Strategy Module
        logger.info("[4/6] Initializing Strategy Module (Signal Generation)")
        await self._setup_strategy()
        
        # Stage 5: Risk Module
        logger.info("[5/10] Initializing Risk Module (Position Sizing & Drawdown Guard)")
        await self._setup_risk()
        
        # Stage 6: Portfolio Risk Module (Optional - Portfolio-level correlation & exposure control)
        logger.info("[6/10] Initializing Portfolio Risk Module (Correlation & Exposure Control)")
        await self._setup_portfolio_risk()
        
        # Stage 7: Regime Capital Module (Optional - Regime-aware capital allocation)
        logger.info("[7/10] Initializing Regime Capital Module (Capital Allocation)")
        await self._setup_regime_capital()
        
        # Stage 8: Execution Module
        logger.info("[8/10] Initializing Execution Module (Broker & Order Manager)")
        await self._setup_execution()
        
        # Stage 9: Execution Reality Module (Optional - Market impact & realism modeling)
        logger.info("[9/10] Initializing Execution Reality Module (Market Impact)")
        await self._setup_execution_reality()
        
        # Stage 10: Optimization Module (Optional - AI-driven performance enhancement)
        logger.info("[10/10] Initializing Optimization Module (Performance Enhancement)")
        await self._setup_optimization()
        
        logger.info("=" * 80)
        logger.info("PLATFORM INITIALIZATION COMPLETE")
        logger.info("=" * 80)
        
        self.logger_mgr.log_event("platform.initialized", {
            "environment": self.state.get_environment().value,
            "exchange": self.config['exchange']['id'],
            "symbol": self.config['trading']['symbol'],
            "strategy": self.strategy.name
        })
    
    async def _setup_core(self):
        """Initialize core module (environment and state)."""
        env_name = self.config.get('environment', 'paper').lower()
        
        # Map string to Environment enum
        env_map = {
            'paper': Environment.PAPER,
            'testnet': Environment.TESTNET,
            'live': Environment.LIVE
        }
        environment = env_map.get(env_name, Environment.PAPER)
        
        # Initialize state
        self.state = TradingState(environment=environment)
        logger.info(f"   ✓ TradingState initialized: {environment.value}")
        
        # Initialize environment manager
        self.env_manager = EnvironmentManager()
        logger.info(f"   ✓ EnvironmentManager initialized")
    
    async def _setup_monitoring(self):
        """Initialize monitoring module (logger and metrics)."""
        # Initialize logger manager
        self.logger_mgr = LoggerManager()
        log_config = self.config.get('monitoring', {})
        self.logger_mgr.configure(
            level=log_config.get('log_level', 'INFO'),
            log_file=log_config.get('log_file', 'logs/trading.log')
        )
        logger.info(f"   ✓ LoggerManager configured")
        
        # Initialize metrics collector
        self.metrics = MetricsCollector()
        logger.info(f"   ✓ MetricsCollector initialized")
    
    async def _setup_data(self):
        """Initialize data module (exchange connector and market feed)."""
        exchange_config = self.config.get('exchange', {})
        env_config = self.env_manager.get_config(self.state.get_environment())
        
        # Determine exchange mode
        if self.state.get_environment() == Environment.PAPER:
            mode = 'paper'
        elif self.state.get_environment() == Environment.TESTNET:
            mode = 'testnet'
        else:
            mode = 'live'
        
        # Initialize exchange connector
        self.exchange = ExchangeConnector(
            exchange_id=exchange_config.get('id', 'binance'),
            api_key=env_config.api_key if env_config else None,
            secret=env_config.api_secret if env_config else None,
            mode=mode
        )
        
        # Connect to exchange
        await self.exchange.connect()
        logger.info(f"   ✓ ExchangeConnector connected: {exchange_config.get('id', 'binance')} ({mode})")
        
        # Initialize market feed
        self.market_feed = MarketFeed(self.exchange)
        logger.info(f"   ✓ MarketFeed initialized")
    
    async def _setup_strategy(self):
        """Initialize strategy module."""
        strategy_config = self.config.get('strategy', {})
        strategy_name = strategy_config.get('name', 'ema_trend')
        
        if strategy_name == 'ema_trend':
            self.strategy = EMATrendStrategy(
                fast_period=strategy_config.get('fast_period', 50),
                slow_period=strategy_config.get('slow_period', 200),
                signal_confidence=strategy_config.get('signal_confidence', 0.7)
            )
        elif strategy_name == 'regime_based':
            # Parse allowed regimes from config
            allowed_regimes_config = strategy_config.get('allowed_regimes', ['trending'])
            allowed_regimes = []
            for regime in allowed_regimes_config:
                if isinstance(regime, str):
                    allowed_regimes.append(MarketRegime[regime.upper()])
                else:
                    allowed_regimes.append(regime)
            
            def _build_strategy_from_config(name: str, params: Dict[str, Any]) -> Optional[BaseStrategy]:
                """Instantiate a strategy from config name and params."""
                lname = name.lower()
                if lname.startswith('ema_trend'):
                    return EMATrendStrategy(
                        fast_period=params.get('fast_period', 50),
                        slow_period=params.get('slow_period', 200),
                        signal_confidence=params.get('signal_confidence', 0.7),
                        name=params.get('name', None) or name
                    )
                if lname in ('continuation', 'continuationstrategy'):
                    return ContinuationStrategy(name=params.get('name', None) or "ContinuationStrategy", **{k: v for k, v in params.items() if k != 'name'})
                if lname in ('fvg', 'fvgstrategy', 'fair_value_gap', 'fairvaluegap'):
                    return FVGStrategy(name=params.get('name', None) or "FVGStrategy", **{k: v for k, v in params.items() if k != 'name'})
                if lname in ('abc', 'abcstrategy'):
                    return ABCStrategy(name=params.get('name', None) or "ABCStrategy", **{k: v for k, v in params.items() if k != 'name'})
                if lname in ('reclamation', 'reclamation_block', 'reclamationblockstrategy'):
                    return ReclamationBlockStrategy(name=params.get('name', None) or "ReclamationBlockStrategy", **{k: v for k, v in params.items() if k != 'name'})
                if lname in ('range', 'range_trading', 'rangetradingstrategy', 'mean_reversion'):
                    return RangeTradingStrategy(name=params.get('name', None) or "RangeTradingStrategy", **{k: v for k, v in params.items() if k != 'name'})
                return None

            # If multi-strategy is enabled, build aggregator; otherwise fall back to single underlying
            multi_cfg = strategy_config.get('multi_strategy', {})
            underlying_strategy: Optional[BaseStrategy] = None

            if multi_cfg.get('enabled', False):
                strategies = []
                weights: Dict[str, float] = {}
                for strat_cfg in multi_cfg.get('strategies', []):
                    if not strat_cfg.get('enabled', True):
                        continue
                    strat_name = strat_cfg.get('name', 'ema_trend')
                    params = strat_cfg.get('params', {})
                    strategy_instance = _build_strategy_from_config(strat_name, params)
                    if strategy_instance is None:
                        logger.warning(f"[Strategy] Unknown strategy in multi_strategy: {strat_name}")
                        continue
                    strategies.append(strategy_instance)
                    weights[strategy_instance.name] = float(strat_cfg.get('weight', 1.0))

                if not strategies:
                    raise ValueError("multi_strategy.enabled=True but no valid strategies were configured")

                adaptive_cfg = multi_cfg.get('adaptive_weighting', {})
                underlying_strategy = MultiStrategyAggregator(
                    strategies=strategies,
                    weights=weights,
                    enable_adaptive_weights=adaptive_cfg.get('enabled', False),
                    min_samples_for_adaptation=adaptive_cfg.get('min_samples', 20),
                    adaptation_learning_rate=adaptive_cfg.get('learning_rate', 0.1),
                )
            else:
                # Create single underlying strategy if specified
                underlying_config = strategy_config.get('underlying_strategy', {})
                underlying_name = underlying_config.get('name', 'ema_trend')
                underlying_strategy = _build_strategy_from_config(underlying_name, underlying_config) or EMATrendStrategy(
                    fast_period=underlying_config.get('fast_period', 50),
                    slow_period=underlying_config.get('slow_period', 200),
                    signal_confidence=underlying_config.get('signal_confidence', 0.7)
                )
            
            # Volatile policy (supports both legacy top-level keys and nested config)
            volatile_policy_cfg = strategy_config.get('volatile_policy', {})
            allow_volatile_trend_override = volatile_policy_cfg.get(
                'enabled', strategy_config.get('allow_volatile_trend_override', False)
            )
            volatile_trend_strength_min = volatile_policy_cfg.get(
                'min_trend_strength', strategy_config.get('volatile_trend_strength_min', 1.0)
            )
            volatile_confidence_scale = volatile_policy_cfg.get(
                'confidence_scale', strategy_config.get('volatile_confidence_scale', 0.5)
            )

            self.strategy = RegimeBasedStrategy(
                underlying_strategy=underlying_strategy,
                allowed_regimes=allowed_regimes,
                regime_confidence_threshold=strategy_config.get('regime_confidence_threshold', 0.5),
                reduce_confidence_in_marginal_regimes=strategy_config.get('reduce_confidence_in_marginal_regimes', True),
                allow_volatile_trend_override=allow_volatile_trend_override,
                volatile_trend_strength_min=volatile_trend_strength_min,
                volatile_confidence_scale=volatile_confidence_scale
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        logger.info(f"   ✓ Strategy initialized: {self.strategy.name}")
    
    async def _setup_risk(self):
        """Initialize risk module (position sizing and drawdown guard)."""
        risk_config = self.config.get('risk', {})
        trading_config = self.config.get('trading', {})
        
        # Initialize position sizer
        sizing_method_str = risk_config.get('sizing_method', 'fixed_risk')
        sizing_method = SizingMethod[sizing_method_str.upper()] if hasattr(SizingMethod, sizing_method_str.upper()) else SizingMethod.FIXED_RISK
        
        initial_equity = trading_config.get('initial_capital', 100000.0)
        
        self.position_sizer = PositionSizer(
            account_equity=initial_equity,
            risk_per_trade=risk_config.get('risk_per_trade', 0.02),
            max_position_size=risk_config.get('max_position_size', initial_equity * 0.1),
            sizing_method=sizing_method,
            use_confidence_scaling=risk_config.get('use_confidence_scaling', True)
        )
        logger.info(f"   ✓ PositionSizer initialized: {sizing_method.value}")
        
        # Initialize drawdown guard
        max_consecutive = risk_config.get('max_consecutive_losses')
        self.drawdown_guard = DrawdownGuard(
            initial_equity=initial_equity,
            max_daily_drawdown=risk_config.get('max_daily_drawdown', 0.05),
            max_total_drawdown=risk_config.get('max_total_drawdown', 0.20),
            max_consecutive_losses=max_consecutive if max_consecutive else None
        )
        logger.info(f"   ✓ DrawdownGuard initialized")
    
    async def _setup_portfolio_risk(self):
        """Initialize portfolio risk manager (portfolio-level correlation & exposure control)."""
        portfolio_config = self.config.get('portfolio_risk', {})
        trading_config = self.config.get('trading', {})
        
        portfolio_enabled = portfolio_config.get('enabled', False)
        
        if not portfolio_enabled:
            logger.info(f"   ✓ PortfolioRiskManager disabled (set portfolio_risk.enabled=True to enable)")
            self.portfolio_risk_manager = None
            return
        
        try:
            initial_equity = trading_config.get('initial_capital', 100000.0)
            
            self.portfolio_risk_manager = PortfolioRiskManager(
                total_equity=initial_equity,
                enabled=True,
                correlation_lookback=portfolio_config.get('correlation_lookback', 100),
                max_pairwise_correlation=portfolio_config.get('max_pairwise_correlation', 0.85),
                max_symbol_exposure_pct=portfolio_config.get('max_symbol_exposure_pct', 0.25),
                max_directional_exposure_pct=portfolio_config.get('max_directional_exposure_pct', 0.60),
                max_regime_exposure_pct=portfolio_config.get('max_regime_exposure_pct', 0.50),
            )
            
            logger.info(f"   ✓ PortfolioRiskManager initialized")
            logger.info(f"   ✓ Max correlation: {portfolio_config.get('max_pairwise_correlation', 0.85):.2f}")
            logger.info(f"   ✓ Max symbol exposure: {portfolio_config.get('max_symbol_exposure_pct', 0.25):.1%}")
            logger.info(f"   ✓ Max directional exposure: {portfolio_config.get('max_directional_exposure_pct', 0.60):.1%}")
            
        except Exception as e:
            logger.error(f"   ✗ Failed to initialize portfolio risk manager: {e}")
            self.portfolio_risk_manager = None
    
    async def _setup_regime_capital(self):
        """Initialize regime capital allocator (regime-aware capital allocation)."""
        regime_config = self.config.get('regime_capital', {})
        trading_config = self.config.get('trading', {})
        
        regime_enabled = regime_config.get('enabled', False)
        
        if not regime_enabled:
            logger.info(f"   ✓ RegimeCapitalAllocator disabled (set regime_capital.enabled=True to enable)")
            self.regime_capital_allocator = None
            return
        
        try:
            initial_equity = trading_config.get('initial_capital', 100000.0)
            
            self.regime_capital_allocator = RegimeCapitalAllocator(
                config=self.config,
                base_equity=initial_equity
            )
            
            logger.info(f"   ✓ RegimeCapitalAllocator initialized")
            logger.info(f"   ✓ Min trades before adjustment: {regime_config.get('min_trades_before_adjustment', 20)}")
            logger.info(f"   ✓ Max allocation shift: {regime_config.get('max_allocation_shift', 0.1):.1%}")
            
        except Exception as e:
            logger.error(f"   ✗ Failed to initialize regime capital allocator: {e}")
            self.regime_capital_allocator = None
    
    async def _setup_execution(self):
        """Initialize execution module (broker and order manager)."""
        exec_config = self.config.get('execution', {})
        
        # Map environment to execution mode
        exec_mode_map = {
            Environment.PAPER: ExecutionMode.PAPER,
            Environment.TESTNET: ExecutionMode.TESTNET,
            Environment.LIVE: ExecutionMode.LIVE
        }
        exec_mode = exec_mode_map.get(self.state.get_environment(), ExecutionMode.PAPER)
        
        # Initialize broker
        self.broker = Broker(
            exchange=self.exchange,
            mode=exec_mode,
            initial_cash=self.config['trading'].get('initial_capital', 100000.0)
        )
        logger.info(f"   ✓ Broker initialized: {exec_mode.value}")
        
        # Initialize order manager
        self.order_manager = OrderManager(
            max_concurrent_orders=exec_config.get('max_concurrent_orders', 5),
            default_max_attempts=exec_config.get('max_retries', 3),
            retry_delay_seconds=exec_config.get('retry_delay_seconds', 5.0)
        )
        self.order_manager.broker = self.broker  # Set broker reference after init
        logger.info(f"   ✓ OrderManager initialized")
        # Live readiness: initialize sim-live executor if configured
        live_cfg = self.config.get('live_readiness', {})
        self.kill_switch = None
        self.health_monitor = None
        if live_cfg.get('enabled', False):
            # Lazy import of kill switch and health monitor
            try:
                from risk.runtime_kill_switch import RuntimeKillSwitch
                from monitoring.runtime_health_monitor import RuntimeHealthMonitor
                self.kill_switch = RuntimeKillSwitch(self.config, initial_equity=self.config['trading'].get('initial_capital', 100000.0))
                self.health_monitor = RuntimeHealthMonitor(self.config)
                logger.info(f"   ✓ Live readiness components initialized")
            except Exception as e:
                logger.error(f"   ✗ Failed to initialize live readiness components: {e}")

        # Wrap broker with SimLiveExecutor if sim_live mode enabled
        if live_cfg.get('enabled', False) and live_cfg.get('sim_live', True):
            self.sim_executor = SimLiveExecutor(
                broker=self.broker,
                shadow_executor=getattr(self, 'shadow_executor', None),
                execution_reality=self.execution_reality_engine,
                kill_switch=self.kill_switch,
                health_monitor=self.health_monitor,
                config=self.config
            )
        else:
            self.sim_executor = None
    
    async def _setup_execution_reality(self):
        """Initialize execution reality module (market impact & realism modeling)."""
        exec_reality_config = self.config.get('execution_reality', {})
        enabled = exec_reality_config.get('enabled', False)
        
        # Initialize engine (always, but disabled by default)
        self.execution_reality_engine = ExecutionRealityEngine(exec_reality_config)
        
        if not enabled:
            logger.info(f"   ✓ Execution Reality disabled (set execution_reality.enabled=True to enable)")
            return
        
        logger.info(f"   ✓ ExecutionRealityEngine initialized")
        logger.info(f"     - Model: {exec_reality_config.get('slippage_model', 'conservative')}")
        logger.info(f"     - Max size vs volume: {exec_reality_config.get('max_size_vs_volume', 0.02):.2%}")
        logger.info(f"     - Min fill probability: {exec_reality_config.get('min_fill_probability', 0.6):.1%}")
    
    async def _setup_optimization(self):
        """Initialize optimization module (AI-driven performance enhancement)."""
        opt_config = self.config.get('optimization', {})
        self.optimization_enabled = opt_config.get('enabled', False)
        
        if not self.optimization_enabled:
            logger.info(f"   ✓ Optimization disabled (set optimization.enabled=True to enable)")
            return
        
        try:
            # Determine base strategy weights
            # If using multi-strategy aggregator, extract weights from it
            base_weights = {}
            if hasattr(self.strategy, 'strategies'):
                # MultiStrategyAggregator
                for strat in self.strategy.strategies:
                    base_weights[strat.name] = self.strategy.weights.get(strat.name, 1.0)
            else:
                # Single strategy
                base_weights[self.strategy.name] = 1.0
            
            # Get risk configuration for optimization bounds
            risk_config = self.config.get('risk', {})
            trading_config = self.config.get('trading', {})
            
            # Initialize optimization engine
            self.optimization_engine = initialize_engine(
                base_weights=base_weights,
                risk_max_drawdown=risk_config.get('max_total_drawdown', 0.20),
                risk_max_position_size=risk_config.get('max_position_size', trading_config.get('initial_capital', 100000.0) * 0.1),
                enable_quality_filtering=opt_config.get('enable_quality_filtering', True),
                enable_adaptive_weighting=opt_config.get('enable_adaptive_weighting', True),
                enable_retirement_management=opt_config.get('enable_retirement_management', True),
            )
            
            # Initialize monitoring
            self.optimization_monitor = get_monitor()
            
            logger.info(f"   ✓ OptimizationEngine initialized with {len(base_weights)} strategies")
            logger.info(f"   ✓ Quality filtering: {opt_config.get('enable_quality_filtering', True)}")
            logger.info(f"   ✓ Adaptive weighting: {opt_config.get('enable_adaptive_weighting', True)}")
            logger.info(f"   ✓ Retirement management: {opt_config.get('enable_retirement_management', True)}")
            
        except Exception as e:
            logger.error(f"   ✗ Failed to initialize optimization: {e}")
            self.optimization_enabled = False
            self.optimization_engine = None
            self.optimization_monitor = None
    
    def _record_trade_entry(
        self,
        symbol: str,
        strategy_name: str,
        side: str,
        entry_price: float,
        quantity: float,
        regime: Optional[str] = None,
    ):
        """
        Record trade entry for later PnL calculation (optimization only).
        
        Args:
            symbol: Trading pair
            strategy_name: Strategy that generated the signal
            side: BUY or SELL
            entry_price: Entry price
            quantity: Trade size
            regime: Market regime at entry
        """
        trade_key = f"{symbol}_{strategy_name}"
        self._open_trades[trade_key] = {
            'symbol': symbol,
            'strategy': strategy_name,
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_timestamp_ms': int(datetime.utcnow().timestamp() * 1000),
            'regime': regime,
        }
        logger.debug(f"[OPTIMIZATION] Trade entry recorded: {trade_key}")
    
    def _record_trade_exit(
        self,
        symbol: str,
        strategy_name: str,
        exit_price: float,
        regime: Optional[str] = None,
    ):
        """
        Record trade exit and calculate PnL for optimization (optimization only).
        
        Args:
            symbol: Trading pair
            strategy_name: Strategy name
            exit_price: Exit price
            regime: Market regime at exit
        """
        trade_key = f"{symbol}_{strategy_name}"
        
        if trade_key not in self._open_trades:
            logger.warning(f"[OPTIMIZATION] No open trade found for {trade_key}")
            return
        
        trade = self._open_trades[trade_key]
        exit_timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
        
        # Calculate PnL
        if trade['side'].upper() == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:  # SELL
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        # Use exit regime if provided, otherwise use entry regime
        trade_regime = regime if regime else trade.get('regime')
        
        # Record trade in performance decomposer
        record_trade(
            strategy_name=trade['strategy'],
            symbol=trade['symbol'],
            pnl=pnl,
            entry_ts=trade['entry_timestamp_ms'],
            exit_ts=exit_timestamp_ms,
            regime=trade_regime,
            size=trade['quantity'],
        )
        
        # REGIME CAPITAL: Record trade result for regime capital allocation
        if self.regime_capital_allocator and self.regime_capital_allocator.enabled:
            if trade_regime:
                self.regime_capital_allocator.record_trade_result(trade_regime, pnl)
                logger.debug(f"[REGIME_CAPITAL] Trade result recorded: {trade_regime} PnL=${pnl:.2f}")
        
        logger.info(
            f"[OPTIMIZATION] Trade completed: {trade['strategy']}/{trade['symbol']} "
            f"PnL=${pnl:.2f} regime={trade_regime}"
        )
        
        # Remove from open trades
        del self._open_trades[trade_key]
    
    async def _optimization_update_loop(self):
        """
        Background task for periodic optimization updates.
        
        Updates adaptive weights, checks strategy health, and manages retirements.
        Runs independently from the main trading loop.
        """
        if not self.optimization_enabled or not self.optimization_engine:
            return
        
        opt_config = self.config.get('optimization', {})
        update_interval = opt_config.get('update_interval_seconds', 3600)
        
        logger.info(f"[OPTIMIZATION] Update loop started (interval: {update_interval}s)")
        
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(update_interval)
                
                if self.shutdown_requested:
                    break
                
                logger.info("[OPTIMIZATION] Running periodic optimization update")
                
                try:
                    # Detect current regime (if available)
                    regime = None
                    if hasattr(self.strategy, 'regime_detector'):
                        # For RegimeBasedStrategy - would need market data
                        # For now, use None to let optimizer use stored regime data
                        pass
                    
                    # Update all optimization components
                    state = self.optimization_engine.update_optimization(regime=regime)
                    
                    # Log significant changes
                    if state.weight_changes_this_period:
                        logger.info(f"[OPTIMIZATION] Weight updates: {state.weight_changes_this_period}")
                    
                    if state.degraded_strategies:
                        logger.warning(f"[OPTIMIZATION] Degraded strategies: {state.degraded_strategies}")
                    
                    if state.suspended_strategies:
                        logger.critical(f"[OPTIMIZATION] Suspended strategies: {state.suspended_strategies}")
                    
                    # Generate health report
                    if self.optimization_monitor:
                        health = self.optimization_monitor.get_health_report()
                        logger.info(
                            f"[OPTIMIZATION] Strategy health: {len(health['active'])} active, "
                            f"{len(health['degraded'])} degraded, {len(health['suspended'])} suspended"
                        )
                    
                except Exception as e:
                    logger.error(f"[OPTIMIZATION] Update failed: {e}", exc_info=True)
        
        except asyncio.CancelledError:
            logger.info("[OPTIMIZATION] Update loop cancelled")
        except Exception as e:
            logger.error(f"[OPTIMIZATION] Update loop error: {e}", exc_info=True)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get current optimization system status.
        
        Returns:
            Dictionary with optimization metrics and health status
        """
        if not self.optimization_enabled or not self.optimization_monitor:
            return {'enabled': False, 'status': 'disabled'}
        
        try:
            health = self.optimization_monitor.get_health_report()
            allocation = self.optimization_monitor.get_allocation_report()
            opt_status = self.optimization_monitor.get_optimization_status()
            
            # Generate alerts
            alerts = []
            if not health['active']:
                alerts.append("CRITICAL: No active strategies")
            if health['suspended']:
                alerts.append(f"WARNING: Suspended strategies: {', '.join(health['suspended'])}")
            if len(health['degraded']) > 2:
                alerts.append(f"ALERT: {len(health['degraded'])} strategies degraded")
            
            # Check for allocation imbalance
            normalized = allocation['normalized_weights']
            if normalized:
                max_weight = max(normalized.values())
                if max_weight > 0.6:
                    alerts.append(f"ALERT: Allocation imbalance (max weight: {max_weight:.1%})")
            
            return {
                'enabled': True,
                'status': opt_status.get('status', 'unknown'),
                'health': health,
                'allocation': allocation,
                'alerts': alerts,
                'open_trades': len(self._open_trades),
            }
        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}", exc_info=True)
            return {'enabled': True, 'status': 'error', 'error': str(e)}
    
    def get_execution_reality_status(self) -> Dict[str, Any]:
        """
        Get current execution reality engine status and diagnostics.
        
        Returns:
            Dictionary with execution metrics and health status
        """
        if not self.execution_reality_engine:
            return {'enabled': False, 'status': 'uninitialized'}
        
        return self.execution_reality_engine.get_diagnostics()
    
    async def _reconcile_positions(self, symbol: str) -> bool:
        """
        Reconcile internal position tracking with broker state.
        
        Args:
            symbol: Trading pair to reconcile
            
        Returns:
            True if positions match, False if desync detected
        """
        if not self.broker:
            return True
        
        try:
            # Get broker's view of position
            broker_position = self.broker.get_position(symbol)
            
            # Check for desyncs in optimization trade tracking
            if self.optimization_enabled:
                trade_key = f"{symbol}_*"  # Check all strategies for this symbol
                tracked_positions = [k for k in self._open_trades.keys() if k.startswith(f"{symbol}_")]
                
                # If broker shows flat but we're tracking open trades
                if abs(broker_position.size) < 1e-8 and tracked_positions:
                    logger.warning(
                        f"[RECONCILIATION] Position desync detected: broker shows flat but tracking "
                        f"{len(tracked_positions)} open trades for {symbol}"
                    )
                    # Clear stale tracking
                    for key in tracked_positions:
                        logger.warning(f"[RECONCILIATION] Clearing stale trade: {key}")
                        del self._open_trades[key]
                    return False
                
                # If broker shows position but we have no tracking
                if abs(broker_position.size) > 1e-8 and not tracked_positions:
                    logger.warning(
                        f"[RECONCILIATION] Position desync detected: broker shows {broker_position.size} "
                        f"but no tracked trades for {symbol}"
                    )
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"[RECONCILIATION] Failed to reconcile {symbol}: {e}")
            return True  # Don't block trading on reconciliation errors
    
    async def _pre_trade_health_check(self, symbol: str) -> tuple[bool, Optional[str]]:
        """
        Perform pre-trade health checks before placing order.
        
        Args:
            symbol: Trading pair to check
            
        Returns:
            (passed, reason) - True if all checks pass, False with reason if any fail
        """
        try:
            # Check 1: Exchange connectivity
            if not self.exchange or not self.exchange.is_connected():
                return False, "Exchange not connected"
            
            # Check 2: Broker is ready
            if not self.broker:
                return False, "Broker not initialized"
            
            # Check 3: Sufficient balance (for live/testnet only)
            if self.state.get_environment() != Environment.PAPER:
                try:
                    equity = self.broker.get_equity()
                    min_equity = self.config['trading'].get('min_equity_required', 100.0)
                    if equity < min_equity:
                        return False, f"Insufficient equity: ${equity:.2f} < ${min_equity:.2f}"
                except Exception as e:
                    logger.warning(f"[HEALTH_CHECK] Could not verify equity: {e}")
            
            # Check 4: Market feed is responding
            if not self.market_feed:
                return False, "Market feed not initialized"
            
            # Check 5: Position reconciliation
            if not await self._reconcile_positions(symbol):
                logger.warning(f"[HEALTH_CHECK] Position desync detected for {symbol} but continuing")
            
            return True, None
            
        except Exception as e:
            logger.error(f"[HEALTH_CHECK] Health check failed: {e}")
            return False, f"Health check error: {str(e)}"
    
    async def run_trading_loop(self):
        """
        Main trading loop - orchestrates data → strategy → risk → execution.
        
        This method contains NO trading logic, only orchestration of module calls.
        """
        logger.info("=" * 80)
        logger.info("STARTING TRADING LOOP")
        logger.info("=" * 80)
        
        self.logger_mgr.log_event("trading.loop_started", {
            "environment": self.state.get_environment().value,
            "symbol": self.config['trading']['symbol']
        })
        
        trading_config = self.config['trading']
        symbol = trading_config['symbol']
        timeframe = trading_config['timeframe']
        
        try:
            while not self.shutdown_requested:
                logger.info("-" * 80)
                logger.info(f"Trading Loop Iteration - {datetime.utcnow().isoformat()}")
                
                # 1. Fetch market data (with timeout protection)
                logger.info(f"[DATA] Fetching {symbol} {timeframe} data")
                try:
                    market_data = await asyncio.wait_for(
                        self.market_feed.fetch(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=trading_config.get('fetch_limit', 500)
                        ),
                        timeout=30.0  # 30 second timeout for market data fetch
                    )
                    logger.info(f"[DATA] Received {len(market_data)} candles")
                except asyncio.TimeoutError:
                    logger.error(f"[DATA] Market data fetch timeout after 30s - skipping iteration")
                    await asyncio.sleep(60)
                    continue
                except Exception as e:
                    logger.error(f"[DATA] Market data fetch failed: {e}")
                    await asyncio.sleep(60)
                    continue
                
                # 1.4. REGIME CAPITAL: Update allocations based on recent performance
                if self.regime_capital_allocator and self.regime_capital_allocator.enabled:
                    self.regime_capital_allocator.update_equity(self.broker.get_equity())
                    self.regime_capital_allocator.update_allocations()
                
                # 1.5. OPTIMIZATION: Check for closed positions and record trade exits
                if self.optimization_enabled:
                    current_position = self.broker.get_position(symbol)
                    current_price = float(market_data.iloc[-1]['close'])
                    
                    # Check if any tracked positions were closed
                    for trade_key in list(self._open_trades.keys()):
                        trade = self._open_trades[trade_key]
                        if trade['symbol'] == symbol:
                            # Position was closed if current position is flat (size ~0)
                            if abs(current_position.size) < 1e-8:
                                # PORTFOLIO RISK: Unregister closed position
                                if self.portfolio_risk_manager and self.portfolio_risk_manager.enabled:
                                    self.portfolio_risk_manager.unregister_position(symbol)
                                
                                # OPTIMIZATION: Record trade exit
                                self._record_trade_exit(
                                    symbol=symbol,
                                    strategy_name=trade['strategy'],
                                    exit_price=current_price,
                                    regime=None,  # Could detect current regime here
                                )
                
                # 2. Generate strategy signal
                logger.info(f"[STRATEGY] Generating signal using {self.strategy.name}")
                current_position = self.broker.get_position(symbol)
                signal_output = self.strategy.generate_signal(
                    market_data=market_data,
                    position_state=current_position
                )
                
                # Normalize regime string to uppercase for consistency
                if hasattr(signal_output, 'metadata') and signal_output.metadata:
                    regime_info = signal_output.metadata.get('regime')
                    if regime_info and isinstance(regime_info, str):
                        signal_output.metadata['regime'] = regime_info.upper()
                
                # Log signal with regime metadata if available
                signal_info = f"Signal: {signal_output.signal.value} (confidence: {signal_output.confidence:.2f})"
                if hasattr(signal_output, 'metadata') and signal_output.metadata:
                    regime_info = signal_output.metadata.get('regime')
                    regime_confidence = signal_output.metadata.get('regime_confidence')
                    if regime_info:
                        signal_info += f" | Regime: {regime_info}"
                        if regime_confidence is not None:
                            signal_info += f" (conf: {regime_confidence:.2f})"
                logger.info(f"[STRATEGY] {signal_info}")
                
                # Log detailed regime metadata if present
                if hasattr(signal_output, 'metadata') and signal_output.metadata:
                    self.logger_mgr.log_event("strategy.signal_generated", {
                        "signal": signal_output.signal.value,
                        "confidence": signal_output.confidence,
                        "metadata": signal_output.metadata
                    })
                
                # 2.5. KILL SWITCH: Early check before processing signals
                if self.kill_switch and self.kill_switch.is_blocked():
                    logger.warning("[KILL_SWITCH] System halted - skipping trade processing")
                    await asyncio.sleep(60)
                    continue
                
                # 2.6. HEALTH CHECK: Verify system readiness before processing signal
                health_passed, health_reason = await self._pre_trade_health_check(symbol)
                if not health_passed:
                    logger.warning(f"[HEALTH_CHECK] Pre-trade check failed: {health_reason}")
                    await asyncio.sleep(60)
                    continue
                
                # 3. Check if signal is actionable
                if signal_output.signal == Signal.HOLD:
                    logger.info("[STRATEGY] HOLD signal - no action required")
                    await asyncio.sleep(60)  # Wait before next iteration
                    continue
                
                # 3.5. OPTIMIZATION: Score and filter signal (pre-trade evaluation)
                if self.optimization_enabled and self.optimization_engine:
                    logger.info(f"[OPTIMIZATION] Evaluating signal quality")
                    
                    # Extract regime information from signal metadata (already normalized to uppercase)
                    regime = None
                    confluence_count = 1
                    if hasattr(signal_output, 'metadata') and signal_output.metadata:
                        regime = signal_output.metadata.get('regime')  # Already uppercase from step 2
                        # Try to infer confluence from metadata
                        if 'confluence_count' in signal_output.metadata:
                            confluence_count = signal_output.metadata['confluence_count']
                    
                    # Determine strategy name
                    strategy_name = self.strategy.name
                    
                    # Score the signal
                    quality, should_execute = self.optimization_engine.score_signal(
                        strategy=strategy_name,
                        symbol=symbol,
                        signal_type=signal_output.signal.value,
                        original_confidence=signal_output.confidence,
                        confluence_count=confluence_count,
                        regime=regime,
                    )
                    
                    if not should_execute:
                        logger.info(
                            f"[OPTIMIZATION] Signal filtered - quality={quality:.3f} below threshold "
                            f"(strategy={strategy_name}, signal={signal_output.signal.value})"
                        )
                        await asyncio.sleep(60)
                        continue
                    
                    # Store quality for position sizing multiplier
                    self._last_signal_quality = quality
                    logger.info(f"[OPTIMIZATION] Signal approved - quality={quality:.3f}")
                
                # 4. Risk Management - Position Sizing
                logger.info(f"[RISK] Calculating position size")
                current_price = float(market_data.iloc[-1]['close'])
                
                # Update position sizer equity
                self.position_sizer.account_equity = self.broker.get_equity()
                
                # Extract stop loss from signal metadata (if available)
                stop_loss_price = None
                if hasattr(signal_output, 'metadata') and signal_output.metadata:
                    stop_loss_price = signal_output.metadata.get('stop_loss')
                
                # Calculate position size
                sizing_result = self.position_sizer.calculate_size(
                    signal={
                        'signal': signal_output.signal.value,
                        'confidence': signal_output.confidence,
                        'metadata': signal_output.metadata if hasattr(signal_output, 'metadata') else {}
                    },
                    current_price=current_price,
                    stop_loss_price=stop_loss_price
                )
                
                # 4.5. OPTIMIZATION: Apply position size multiplier
                original_position_size = sizing_result.position_size
                if self.optimization_enabled and self.optimization_engine and sizing_result.approved:
                    # Get regime for multiplier calculation (already normalized to uppercase)
                    regime = None
                    if hasattr(signal_output, 'metadata') and signal_output.metadata:
                        regime = signal_output.metadata.get('regime')
                    
                    # Get signal quality (already calculated above)
                    quality = getattr(self, '_last_signal_quality', signal_output.confidence)
                    
                    # Get position size multiplier
                    multiplier = self.optimization_engine.get_position_size_multiplier(
                        strategy=strategy_name,
                        signal_quality=quality,
                        regime=regime,
                    )
                    
                    # Apply multiplier to position size
                    sizing_result.position_size = original_position_size * multiplier
                    
                    logger.info(
                        f"[OPTIMIZATION] Position size adjusted: "
                        f"{original_position_size:.6f} → {sizing_result.position_size:.6f} "
                        f"({multiplier:.2f}x multiplier)"
                    )
                
                logger.info(f"[RISK] Position size: {sizing_result.position_size:.6f} (approved: {sizing_result.approved})")
                
                if not sizing_result.approved:
                    logger.warning(f"[RISK] Position sizing rejected: {sizing_result.rejection_reason}")
                    await asyncio.sleep(60)
                    continue
                
                # 5. Risk Management - Drawdown Check
                logger.info(f"[RISK] Checking drawdown limits")
                self.drawdown_guard.update_equity(self.broker.get_equity())
                veto_decision = self.drawdown_guard.check_trade()
                logger.info(f"[RISK] Drawdown check: {veto_decision.approved}")
                
                if not veto_decision.approved:
                    logger.warning(f"[RISK] Trade vetoed by drawdown guard: {veto_decision.veto_reason.value if veto_decision.veto_reason else 'unknown'}")
                    self.logger_mgr.log_event("risk.trade_vetoed", {
                        "reason": veto_decision.veto_reason.value if veto_decision.veto_reason else "unknown",
                        "metrics": veto_decision.current_metrics.to_dict() if veto_decision.current_metrics else {}
                    })
                    await asyncio.sleep(60)
                    continue
                
                # 5.5. PORTFOLIO RISK: Validate position with correlation & exposure controls
                if self.portfolio_risk_manager and self.portfolio_risk_manager.enabled:
                    logger.info(f"[PORTFOLIO_RISK] Validating position")
                    
                    # Update market data for correlation calculation
                    self.portfolio_risk_manager.update_market_data(symbol, market_data)
                    
                    # Update equity
                    self.portfolio_risk_manager.update_equity(self.broker.get_equity())
                    
                    # Extract regime from signal metadata (already normalized to uppercase)
                    regime = signal_output.metadata.get('regime') if hasattr(signal_output, 'metadata') and signal_output.metadata else None
                    
                    # Validate new position
                    allowed, reason, portfolio_multiplier = self.portfolio_risk_manager.validate_new_position(
                        symbol=symbol,
                        side='long' if signal_output.signal.value.upper() == 'BUY' else 'short',
                        proposed_size=sizing_result.position_size,
                        regime=regime,
                        strategy=strategy_name if 'strategy_name' in locals() else self.strategy.name,
                    )
                    
                    if not allowed:
                        logger.warning(
                            f"[PORTFOLIO_RISK] Trade blocked: {symbol} "
                            f"reason={reason}"
                        )
                        await asyncio.sleep(60)
                        continue
                    
                    # Apply portfolio multiplier
                    if portfolio_multiplier < 1.0:
                        original_size = sizing_result.position_size
                        sizing_result.position_size = original_size * portfolio_multiplier
                        
                        logger.info(
                            f"[PORTFOLIO_RISK] Position size reduced: "
                            f"{original_size:.6f} → {sizing_result.position_size:.6f} "
                            f"({portfolio_multiplier:.2f}x multiplier) reason={reason}"
                        )
                
                # 5.6. REGIME CAPITAL: Apply regime-aware capital allocation multiplier
                if self.regime_capital_allocator and self.regime_capital_allocator.enabled:
                    logger.info(f"[REGIME_CAPITAL] Applying regime capital allocation")
                    
                    # Extract regime from signal metadata (already normalized to uppercase)
                    regime = None
                    if hasattr(signal_output, 'metadata') and signal_output.metadata:
                        regime = signal_output.metadata.get('regime')
                    
                    if regime and regime in ['TRENDING', 'RANGING', 'VOLATILE']:
                        regime_multiplier = self.regime_capital_allocator.get_regime_multiplier(regime)
                        
                        if regime_multiplier < 1.0:
                            original_size = sizing_result.position_size
                            sizing_result.position_size = original_size * regime_multiplier
                            
                            logger.info(
                                f"[REGIME_CAPITAL] Position size reduced: "
                                f"{original_size:.6f} → {sizing_result.position_size:.6f} "
                                f"({regime_multiplier:.2f}x multiplier) regime={regime}"
                            )
                        else:
                            logger.debug(
                                f"[REGIME_CAPITAL] No reduction applied: {regime} has full allocation"
                            )
                    else:
                        logger.debug(f"[REGIME_CAPITAL] Unknown or missing regime: {regime}")
                
                # 5.7. EXECUTION REALITY: Evaluate execution feasibility and apply market constraints
                if self.execution_reality_engine and self.execution_reality_engine.enabled:
                    logger.info(f"[EXECUTION_REALITY] Evaluating order feasibility")
                    
                    # Create market snapshot from current data
                    candle = market_data.iloc[-1]
                    market_snapshot = {
                        'close': float(candle['close']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'volume': float(candle['volume']) if 'volume' in candle else 1e10,
                    }
                    
                    # Calculate ATR percent if available
                    if 'volume' in candle and len(market_data) > 1:
                        # Simple volatility proxy: (high - low) / close
                        atr_percent = (float(candle['high']) - float(candle['low'])) / float(candle['close']) * 100
                        market_snapshot['atr_percent'] = atr_percent
                    
                    # Evaluate order with execution reality engine
                    execution_eval = self.execution_reality_engine.evaluate_order(
                        symbol=symbol,
                        side=signal_output.signal.value.lower(),
                        requested_size=sizing_result.position_size,
                        market_snapshot=market_snapshot
                    )
                    
                    # Extract results
                    approved_size = execution_eval['approved_size']
                    slippage_bps = execution_eval['expected_slippage_bps']
                    fill_probability = execution_eval['fill_probability']
                    execution_risk = execution_eval['execution_risk']
                    rejection_reason = execution_eval['rejection_reason']
                    
                    # Log evaluation results
                    if approved_size < sizing_result.position_size:
                        if approved_size == 0:
                            logger.warning(
                                f"[EXECUTION_REALITY] Order rejected: {rejection_reason} | "
                                f"fill_prob={fill_probability:.1%}, slippage={slippage_bps:.0f}bps, "
                                f"risk={execution_risk}"
                            )
                            # Skip trade if execution quality too poor
                            await asyncio.sleep(60)
                            continue
                        else:
                            original_size = sizing_result.position_size
                            sizing_result.position_size = approved_size
                            logger.info(
                                f"[EXECUTION_REALITY] Position size reduced: "
                                f"{original_size:.6f} → {approved_size:.6f} | "
                                f"fill_prob={fill_probability:.1%}, slippage={slippage_bps:.0f}bps, "
                                f"risk={execution_risk}"
                            )
                    else:
                        logger.debug(
                            f"[EXECUTION_REALITY] Order approved: "
                            f"fill_prob={fill_probability:.1%}, slippage={slippage_bps:.0f}bps, "
                            f"risk={execution_risk}"
                        )
                
                # 5.8. FINAL VALIDATION: Verify position size after all multipliers
                risk_config = self.config.get('risk', {})
                max_position_size = risk_config.get('max_position_size', self.broker.get_equity() * 0.1)
                min_position_size = current_price * 0.00001  # Minimum notional value
                
                if sizing_result.position_size <= 0:
                    logger.warning(f"[VALIDATION] Position size is zero or negative: {sizing_result.position_size}")
                    await asyncio.sleep(60)
                    continue
                
                if sizing_result.position_size < min_position_size:
                    logger.warning(
                        f"[VALIDATION] Position size too small: {sizing_result.position_size:.8f} < {min_position_size:.8f} "
                        f"(min notional)"
                    )
                    await asyncio.sleep(60)
                    continue
                
                if sizing_result.position_size > max_position_size:
                    logger.warning(
                        f"[VALIDATION] Position size exceeds max: {sizing_result.position_size:.6f} > {max_position_size:.6f} "
                        f"(capping to max)"
                    )
                    sizing_result.position_size = max_position_size
                
                logger.info(f"[VALIDATION] Final position size: {sizing_result.position_size:.6f} (approved)")
                
                # 6. Execution - Place Order
                logger.info(f"[EXECUTION] Placing {signal_output.signal.value} order")
                # Use SimLiveExecutor when available to enforce kill switch and sim-live routing
                if getattr(self, 'sim_executor', None):
                    order_result = await self.sim_executor.place_market_order(
                        symbol=symbol,
                        side=signal_output.signal.value.lower(),
                        quantity=sizing_result.position_size
                    )
                else:
                    order_result = await self.broker.place_market_order(
                        symbol=symbol,
                        side=signal_output.signal.value.lower(),
                        quantity=sizing_result.position_size
                    )
                
                if order_result.get('status') == 'filled':
                    logger.info(f"[EXECUTION] Order filled: {order_result.get('fill_price')} x {order_result.get('filled_quantity')}")
                    self.logger_mgr.log_trade({
                        "symbol": symbol,
                        "side": signal_output.signal.value,
                        "price": order_result.get('fill_price'),
                        "quantity": order_result.get('filled_quantity'),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Update drawdown guard with new equity after trade execution
                    if self.drawdown_guard:
                        self.drawdown_guard.update_equity(self.broker.get_equity())
                        logger.debug(f"[RISK] Drawdown guard equity updated post-trade: ${self.broker.get_equity():.2f}")
                    
                    # EXECUTION_REALITY: Record actual execution result for diagnostics
                    if self.execution_reality_engine and self.execution_reality_engine.enabled:
                        # Calculate actual slippage
                        expected_price = current_price
                        actual_price = order_result.get('fill_price', current_price)
                        actual_slippage_bps = abs(actual_price - expected_price) / expected_price * 10000 if expected_price > 0 else 0
                        
                        self.execution_reality_engine.record_execution_result(
                            symbol=symbol,
                            requested_size=sizing_result.position_size,
                            filled_size=order_result.get('filled_quantity'),
                            slippage_bps=actual_slippage_bps,
                            latency_ms=0  # Would come from broker latency tracking
                        )
                    
                    # PORTFOLIO RISK: Register position
                    if self.portfolio_risk_manager and self.portfolio_risk_manager.enabled:
                        regime = signal_output.metadata.get('regime') if hasattr(signal_output, 'metadata') and signal_output.metadata else None
                        
                        self.portfolio_risk_manager.register_position(
                            symbol=symbol,
                            side='long' if signal_output.signal.value.upper() == 'BUY' else 'short',
                            size=order_result.get('filled_quantity'),
                            regime=regime,
                            strategy=strategy_name if 'strategy_name' in locals() else self.strategy.name,
                        )
                    
                    # OPTIMIZATION: Track trade entry for future PnL calculation
                    if self.optimization_enabled:
                        self._record_trade_entry(
                            symbol=symbol,
                            strategy_name=strategy_name if 'strategy_name' in locals() else self.strategy.name,
                            side=signal_output.signal.value,
                            entry_price=order_result.get('fill_price'),
                            quantity=order_result.get('filled_quantity'),
                            regime=regime if 'regime' in locals() else None,
                        )
                else:
                    logger.warning(f"[EXECUTION] Order not filled: {order_result.get('status')}")
                
                # Wait before next iteration
                await asyncio.sleep(60)

                # If CLI requested single-run mode, break after first iteration
                if getattr(self, 'run_once', False):
                    logger.info("Run-once flag set - exiting trading loop after single iteration")
                    self.shutdown_requested = True
                    break
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
            self.logger_mgr.log_error("trading.loop_error", exc_info=True)
            raise
    
    async def shutdown(self):
        """
        Graceful shutdown of all modules.
        
        Shutdown Order: Execution → Risk → Strategy → Data → Monitoring → Core
        """
        logger.info("=" * 80)
        logger.info("INITIATING GRACEFUL SHUTDOWN")
        logger.info("=" * 80)
        
        self.shutdown_requested = True
        
        try:
            # Log shutdown event
            if self.logger_mgr:
                self.logger_mgr.log_event("platform.shutdown_started", {
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Stage 1: Execution Module
            logger.info("[1/6] Shutting down Execution Module")
            if self.order_manager:
                # Cancel pending orders
                stats = self.order_manager.get_stats()
                logger.info(f"   ✓ OrderManager stats: {stats.completed_orders} completed, {stats.pending_orders} pending")
            if self.broker:
                logger.info(f"   ✓ Broker final equity: ${self.broker.get_equity():.2f}")
            
            # Stage 2: Risk Module
            logger.info("[2/6] Shutting down Risk Module")
            if self.drawdown_guard:
                metrics = self.drawdown_guard.get_current_metrics()
                logger.info(f"   ✓ Final drawdown metrics: {metrics}")
            
            # Stage 3: Strategy Module
            logger.info("[3/6] Shutting down Strategy Module")
            if self.strategy:
                logger.info(f"   ✓ Strategy: {self.strategy.name}")
            
            # Stage 4: Data Module
            logger.info("[4/6] Shutting down Data Module")
            if self.exchange:
                await self.exchange.disconnect()
                logger.info(f"   ✓ Exchange disconnected")
            
            # Stage 5: Monitoring Module
            logger.info("[5/6] Shutting down Monitoring Module")
            if self.metrics:
                final_metrics = self.metrics.get_overall_metrics()
                logger.info(f"   ✓ Final metrics: {final_metrics.get('total_trades', 0)} trades")
            
            # Log optimization final state
            if self.optimization_enabled and self.optimization_monitor:
                logger.info("[5.5/6] Optimization Final Report")
                try:
                    health = self.optimization_monitor.get_health_report()
                    alloc = self.optimization_monitor.get_allocation_report()
                    
                    logger.info(f"   ✓ Strategy Health:")
                    logger.info(f"      - Active: {health['active']}")
                    logger.info(f"      - Degraded: {health['degraded']}")
                    logger.info(f"      - Suspended: {health['suspended']}")
                    logger.info(f"   ✓ Final Allocation: {alloc['normalized_weights']}")
                except Exception as e:
                    logger.error(f"   ✗ Failed to generate optimization report: {e}")
            
            # Stage 6: Core Module
            logger.info("[6/6] Shutting down Core Module")
            if self.state:
                logger.info(f"   ✓ Final state: {self.state.get_environment().value}")
            
            logger.info("=" * 80)
            logger.info("SHUTDOWN COMPLETE")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    async def run(self):
        """
        Main entry point - setup, run trading loop, and shutdown.
        """
        # Ensure optimization_task is defined for finally block even if setup fails
        optimization_task = None
        try:
            await self.setup()
            
            # Start optimization update loop as background task (if enabled)
            if self.optimization_enabled:
                optimization_task = asyncio.create_task(self._optimization_update_loop())
                logger.info("[OPTIMIZATION] Background update task started")
            
            # Run main trading loop
            await self.run_trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            # Cancel optimization task if running
            if optimization_task and not optimization_task.done():
                optimization_task.cancel()
                try:
                    await optimization_task
                except asyncio.CancelledError:
                    pass
            
            await self.shutdown()


# Global platform instance for signal handler
platform_instance = None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    if platform_instance:
        platform_instance.shutdown_requested = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Platform CLI")
    parser.add_argument('-c', '--config', default='core/config.yaml', help='Path to config YAML')
    parser.add_argument('-e', '--environment', choices=['paper', 'testnet', 'live'], help='Override environment')
    parser.add_argument('--optimization', dest='optimization', action='store_true', help='Enable optimization')
    parser.add_argument('--no-optimization', dest='optimization', action='store_false', help='Disable optimization')
    parser.set_defaults(optimization=None)
    parser.add_argument('--exchange-id', dest='exchange_id', help='Override exchange id')
    parser.add_argument('--log-level', dest='log_level', help='Override log level (DEBUG/INFO/WARNING/ERROR)')
    parser.add_argument('--once', dest='once', action='store_true', help='Run a single trading loop iteration')

    args = parser.parse_args()

    # Prepare overrides to apply after config load
    overrides: Dict[str, Any] = {}
    if args.environment:
        overrides['environment'] = args.environment
    if args.optimization is not None:
        overrides['optimization_enabled'] = bool(args.optimization)
    if args.exchange_id:
        overrides['exchange_id'] = args.exchange_id
    if args.log_level:
        overrides['log_level'] = args.log_level

    try:
        # Create platform instance with parsed CLI options
        platform_instance = TradingPlatform(config_path=args.config, overrides=overrides, run_once=bool(args.once))

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run platform
        asyncio.run(platform_instance.run())
    except KeyboardInterrupt:
        logger.info("Shutdown via keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)

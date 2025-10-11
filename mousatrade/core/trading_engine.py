"""
Main Trading Engine - Brain of the Bot
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from enum import Enum
import asyncio
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from ..utils.decorators import retry, timer, thread_safe
from ..strategies.base_strategy import BaseStrategy
from ..brokers.base_broker import BaseBroker


class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float
    confidence: float
    price: float
    timestamp: datetime
    strategy: str
    metadata: Dict[str, Any] = None


@dataclass
class Order:
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # BUY, SELL
    quantity: float
    price: float
    timestamp: datetime
    status: str = "PENDING"
    filled_quantity: float = 0.0
    average_price: float = 0.0


class TradingEngine:
    """
    Main Trading Engine - Orchestrates all trading activities
    """
    
    def __init__(self, 
                 broker: BaseBroker,
                 initial_balance: float = 10000,
                 trading_mode: TradingMode = TradingMode.PAPER,
                 max_workers: int = 10):
        
        self.broker = broker
        self.initial_balance = initial_balance
        self.trading_mode = trading_mode
        self.max_workers = max_workers
        
        # Core components
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_orders: Dict[str, Order] = {}
        self.portfolio = {}
        
        # State management
        self.is_running = False
        self.is_paused = False
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(initial_balance)
        
        # Event system
        self.event_dispatcher = EventDispatcher()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = asyncio.new_event_loop()
        
        self.logger = self._setup_logging()
        self._setup_event_handlers()
    
    def _setup_logging(self):
        """Setup trading engine logging"""
        logger = logging.getLogger("trading_engine")
        return logger
    
    def _setup_event_handlers(self):
        """Setup event handlers for trading events"""
        self.event_dispatcher.register_handler("order_filled", self._on_order_filled)
        self.event_dispatcher.register_handler("order_cancelled", self._on_order_cancelled)
        self.event_dispatcher.register_handler("market_data", self._on_market_data)
        self.event_dispatcher.register_handler("strategy_signal", self._on_strategy_signal)
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a trading strategy to the engine"""
        strategy_id = f"{strategy.name}_{len(self.strategies)}"
        self.strategies[strategy_id] = strategy
        self.logger.info(f"Strategy added: {strategy_id}")
        
        # Connect strategy to event system
        self.event_dispatcher.register_handler(
            f"strategy_signal_{strategy_id}", 
            lambda signal: self._on_strategy_signal(signal)
        )
    
    def remove_strategy(self, strategy_id: str):
        """Remove a trading strategy"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.logger.info(f"Strategy removed: {strategy_id}")
    
    @retry(max_attempts=3, delay=1.0)
    async def execute_order(self, order: Order) -> bool:
        """Execute a trading order with retry logic"""
        try:
            if self.trading_mode == TradingMode.BACKTEST:
                return await self._execute_backtest_order(order)
            elif self.trading_mode == TradingMode.PAPER:
                return await self._execute_paper_order(order)
            else:  # LIVE
                return await self._execute_live_order(order)
                
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            return False
    
    async def _execute_live_order(self, order: Order) -> bool:
        """Execute order in live trading mode"""
        try:
            # Convert to broker order format
            broker_order = {
                'symbol': order.symbol,
                'side': order.side,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price
            }
            
            # Execute via broker
            result = await self.broker.place_order(broker_order)
            
            if result['success']:
                order.status = "FILLED"
                order.filled_quantity = order.quantity
                order.average_price = result.get('average_price', order.price)
                
                # Emit order filled event
                self.event_dispatcher.emit("order_filled", order)
                return True
            else:
                order.status = "REJECTED"
                return False
                
        except Exception as e:
            self.logger.error(f"Live order execution failed: {e}")
            order.status = "ERROR"
            return False
    
    async def _execute_paper_order(self, order: Order) -> bool:
        """Execute order in paper trading mode"""
        try:
            # Simulate order execution with realistic delays
            await asyncio.sleep(0.1)  # Simulate network latency
            
            # Simulate partial fills, slippage, etc.
            fill_ratio = 1.0  # Could be random for simulation
            slippage = 0.001  # 0.1% slippage
            
            order.status = "FILLED"
            order.filled_quantity = order.quantity * fill_ratio
            order.average_price = order.price * (1 + slippage)
            
            # Update portfolio in paper trading
            self._update_paper_portfolio(order)
            
            # Emit order filled event
            self.event_dispatcher.emit("order_filled", order)
            return True
            
        except Exception as e:
            self.logger.error(f"Paper order execution failed: {e}")
            return False
    
    async def _execute_backtest_order(self, order: Order) -> bool:
        """Execute order in backtest mode"""
        # In backtest, orders are always filled immediately
        order.status = "FILLED"
        order.filled_quantity = order.quantity
        order.average_price = order.price
        
        self.event_dispatcher.emit("order_filled", order)
        return True
    
    def _update_paper_portfolio(self, order: Order):
        """Update paper trading portfolio"""
        symbol = order.symbol
        if symbol not in self.portfolio:
            self.portfolio[symbol] = {'quantity': 0, 'average_price': 0}
        
        position = self.portfolio[symbol]
        
        if order.side == "BUY":
            new_quantity = position['quantity'] + order.filled_quantity
            if new_quantity > 0:
                position['average_price'] = (
                    (position['quantity'] * position['average_price']) +
                    (order.filled_quantity * order.average_price)
                ) / new_quantity
            position['quantity'] = new_quantity
            
        elif order.side == "SELL":
            position['quantity'] -= order.filled_quantity
    
    async def process_market_data(self, symbol: str, data: pd.DataFrame):
        """Process incoming market data"""
        if self.is_paused or not self.is_running:
            return
        
        # Emit market data event
        self.event_dispatcher.emit("market_data", {
            'symbol': symbol,
            'data': data,
            'timestamp': datetime.now()
        })
        
        # Generate signals from all strategies
        signals = await self._generate_signals(symbol, data)
        
        # Process signals
        for signal in signals:
            await self._process_signal(signal)
    
    async def _generate_signals(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from all strategies"""
        signals = []
        
        # Run strategies in parallel
        tasks = []
        for strategy_id, strategy in self.strategies.items():
            task = self.executor.submit(
                self._run_strategy, strategy, symbol, data.copy()
            )
            tasks.append((strategy_id, task))
        
        # Collect results
        for strategy_id, task in tasks:
            try:
                signal_data = task.result(timeout=30)  # 30 second timeout
                if signal_data and signal_data.get('position', 0) != 0:
                    signal = TradingSignal(
                        symbol=symbol,
                        action="BUY" if signal_data['position'] > 0 else "SELL",
                        strength=abs(signal_data['position']),
                        confidence=signal_data.get('confidence', 0.5),
                        price=data['close'].iloc[-1],
                        timestamp=datetime.now(),
                        strategy=strategy_id,
                        metadata=signal_data
                    )
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Strategy {strategy_id} failed: {e}")
        
        return signals
    
    def _run_strategy(self, strategy: BaseStrategy, symbol: str, data: pd.DataFrame):
        """Run strategy in thread pool"""
        try:
            signals = strategy.generate_signals(data)
            return signals.iloc[-1].to_dict() if not signals.empty else None
        except Exception as e:
            self.logger.error(f"Strategy execution error: {e}")
            return None
    
    async def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        try:
            # Risk management check
            if not await self._risk_management_check(signal):
                self.logger.warning(f"Risk check failed for signal: {signal}")
                return
            
            # Position sizing
            quantity = await self._calculate_position_size(signal)
            if quantity <= 0:
                return
            
            # Create order
            order = Order(
                order_id=f"order_{datetime.now().timestamp()}",
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                side=signal.action,
                quantity=quantity,
                price=signal.price,
                timestamp=datetime.now()
            )
            
            # Execute order
            success = await self.execute_order(order)
            
            if success:
                self.logger.info(f"Order executed: {order}")
                self.active_orders[order.order_id] = order
            else:
                self.logger.error(f"Order failed: {order}")
                
        except Exception as e:
            self.logger.error(f"Signal processing failed: {e}")
    
    async def _risk_management_check(self, signal: TradingSignal) -> bool:
        """Risk management validation"""
        # Implement risk checks here
        # - Portfolio concentration
        # - Maximum drawdown
        # - Correlation limits
        # - Volatility checks
        
        # For now, basic check
        return signal.confidence > 0.3 and signal.strength > 0.1
    
    async def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on risk management"""
        # Implement position sizing logic
        # - Kelly criterion
        # - Fixed fractional
        # - Volatility-adjusted
        
        # Simple implementation for now
        account_balance = self.performance_tracker.get_current_balance()
        risk_per_trade = 0.02  # 2% risk per trade
        position_size = (account_balance * risk_per_trade) / signal.price
        
        return round(position_size, 6)  # Round to 6 decimal places
    
    def start(self):
        """Start the trading engine"""
        self.is_running = True
        self.is_paused = False
        self.logger.info("Trading engine started")
        
        # Start performance tracking
        self.performance_tracker.start()
    
    def pause(self):
        """Pause trading"""
        self.is_paused = True
        self.logger.info("Trading engine paused")
    
    def resume(self):
        """Resume trading"""
        self.is_paused = False
        self.logger.info("Trading engine resumed")
    
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.performance_tracker.stop()
        self.logger.info("Trading engine stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return self.performance_tracker.generate_report()
    
    def _on_order_filled(self, order: Order):
        """Handle order filled event"""
        self.logger.info(f"Order filled: {order}")
        
        # Update performance tracker
        self.performance_tracker.record_trade(order)
        
        # Update portfolio
        if order.symbol not in self.portfolio:
            self.portfolio[order.symbol] = {'quantity': 0, 'average_price': 0}
    
    def _on_order_cancelled(self, order: Order):
        """Handle order cancelled event"""
        self.logger.info(f"Order cancelled: {order}")
        
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
    
    def _on_market_data(self, data: Dict[str, Any]):
        """Handle market data event"""
        # Update performance tracker with latest prices
        symbol = data['symbol']
        price = data['data']['close'].iloc[-1]
        self.performance_tracker.update_market_price(symbol, price)
    
    def _on_strategy_signal(self, signal: TradingSignal):
        """Handle strategy signal event"""
        # Signals are processed in process_market_data
        pass
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status report"""
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'trading_mode': self.trading_mode.value,
            'active_strategies': len(self.strategies),
            'active_orders': len(self.active_orders),
            'portfolio_size': len(self.portfolio),
            'current_balance': self.performance_tracker.get_current_balance(),
            'total_trades': self.performance_tracker.get_total_trades()
        }

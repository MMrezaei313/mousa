"""
Real-time Performance Tracking for Trading Engine
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import threading
import time


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    fees: float = 0.0


@dataclass
class PerformanceMetrics:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win: float
    largest_loss: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float


class PerformanceTracker:
    """
    Tracks trading performance in real-time
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.portfolio_value = initial_balance
        
        # Data storage
        self.trades: List[TradeRecord] = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.daily_returns = []
        
        # Real-time tracking
        self.open_positions: Dict[str, Dict] = {}
        self.market_prices: Dict[str, float] = {}
        
        # Performance cache
        self._metrics_cache = None
        self._cache_timestamp = None
        self.cache_ttl = 60  # seconds
        
        # Threading
        self.lock = threading.RLock()
        self.is_running = False
        self.update_interval = 5  # seconds
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup performance tracker logging"""
        logger = logging.getLogger("performance_tracker")
        return logger
    
    def start(self):
        """Start real-time performance tracking"""
        self.is_running = True
        self._start_background_updater()
        self.logger.info("Performance tracker started")
    
    def stop(self):
        """Stop performance tracking"""
        self.is_running = False
        self.logger.info("Performance tracker stopped")
    
    def _start_background_updater(self):
        """Start background thread for real-time updates"""
        def updater():
            while self.is_running:
                try:
                    self._update_portfolio_value()
                    self._update_equity_curve()
                    self._update_drawdown()
                    time.sleep(self.update_interval)
                except Exception as e:
                    self.logger.error(f"Background updater error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=updater, daemon=True)
        thread.start()
    
    def record_trade(self, order: Any):
        """Record a completed trade"""
        with self.lock:
            # This is a simplified implementation
            # In practice, you'd have more detailed trade tracking
            
            # Calculate PnL for the trade
            # This would depend on your position tracking logic
            
            trade = TradeRecord(
                trade_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.filled_quantity,
                entry_price=order.average_price,  # Simplified
                exit_price=order.average_price,   # Simplified
                pnl=0.0,  # Would calculate based on position
                pnl_percent=0.0,
                entry_time=order.timestamp,
                exit_time=order.timestamp,
                strategy="unknown"
            )
            
            self.trades.append(trade)
            self._invalidate_cache()
    
    def update_market_price(self, symbol: str, price: float):
        """Update market price for a symbol"""
        with self.lock:
            self.market_prices[symbol] = price
            self._invalidate_cache()
    
    def _update_portfolio_value(self):
        """Update current portfolio value"""
        with self.lock:
            # Calculate portfolio value based on open positions and market prices
            position_value = 0.0
            
            for symbol, position in self.open_positions.items():
                if symbol in self.market_prices:
                    market_price = self.market_prices[symbol]
                    position_value += position['quantity'] * market_price
            
            self.portfolio_value = self.current_balance + position_value
    
    def _update_equity_curve(self):
        """Update equity curve"""
        with self.lock:
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': self.portfolio_value,
                'balance': self.current_balance
            })
            
            # Keep only last 10,000 points
            if len(self.equity_curve) > 10000:
                self.equity_curve.pop(0)
    
    def _update_drawdown(self):
        """Update drawdown calculations"""
        if not self.equity_curve:
            return
        
        with self.lock:
            current_equity = self.equity_curve[-1]['equity']
            peak_equity = max(point['equity'] for point in self.equity_curve)
            drawdown = (current_equity - peak_equity) / peak_equity
            
            self.drawdown_curve.append({
                'timestamp': datetime.now(),
                'drawdown': drawdown,
                'current_equity': current_equity,
                'peak_equity': peak_equity
            })
            
            # Keep only last 10,000 points
            if len(self.drawdown_curve) > 10000:
                self.drawdown_curve.pop(0)
    
    def _invalidate_cache(self):
        """Invalidate performance metrics cache"""
        self._metrics_cache = None
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        with self.lock:
            # Use cache if available and fresh
            if (self._metrics_cache and 
                self._cache_timestamp and 
                (datetime.now() - self._cache_timestamp).total_seconds() < self.cache_ttl):
                return self._metrics_cache
            
            # Calculate metrics from trades and equity curve
            if not self.trades:
                return self._get_empty_metrics()
            
            # Extract trade PnLs
            trade_pnls = [trade.pnl for trade in self.trades]
            trade_returns = [trade.pnl_percent for trade in self.trades]
            
            # Basic metrics
            total_trades = len(self.trades)
            winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
            losing_trades = len([pnl for pnl in trade_pnls if pnl < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(trade_pnls)
            total_return = total_pnl / self.initial_balance
            
            # Risk-adjusted metrics
            returns_series = pd.Series(trade_returns)
            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0
            annual_return = total_return * (252 / len(returns_series)) if len(returns_series) > 0 else 0
            
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Drawdown from equity curve
            equity_values = [point['equity'] for point in self.equity_curve]
            if equity_values:
                running_max = pd.Series(equity_values).cummax()
                drawdowns = (pd.Series(equity_values) - running_max) / running_max
                max_drawdown = drawdowns.min()
            else:
                max_drawdown = 0
            
            # Profit factor
            gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Additional metrics
            avg_winning_trade = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if winning_trades > 0 else 0
            avg_losing_trade = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if losing_trades > 0 else 0
            largest_win = max(trade_pnls) if trade_pnls else 0
            largest_loss = min(trade_pnls) if trade_pnls else 0
            
            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio (downside risk only)
            downside_returns = returns_series[returns_series < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
            sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_winning_trade=avg_winning_trade,
                avg_losing_trade=avg_losing_trade,
                largest_win=largest_win,
                largest_loss=largest_loss,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio
            )
            
            # Update cache
            self._metrics_cache = metrics
            self._cache_timestamp = datetime.now()
            
            return metrics
    
    def _get_empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades available"""
        return PerformanceMetrics(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_winning_trade=0.0,
            avg_losing_trade=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            volatility=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0
        )
    
    def get_current_balance(self) -> float:
        """Get current account balance"""
        with self.lock:
            return self.current_balance
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        with self.lock:
            return self.portfolio_value
    
    def get_total_trades(self) -> int:
        """Get total number of trades"""
        with self.lock:
            return len(self.trades)
    
    def get_recent_trades(self, limit: int = 20) -> List[TradeRecord]:
        """Get recent trades"""
        with self.lock:
            return self.trades[-limit:]
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        with self.lock:
            return pd.DataFrame(self.equity_curve)
    
    def get_drawdown_curve(self) -> pd.DataFrame:
        """Get drawdown curve as DataFrame"""
        with self.lock:
            return pd.DataFrame(self.drawdown_curve)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        metrics = self.calculate_performance_metrics()
        
        report = {
            'summary': {
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'portfolio_value': self.portfolio_value,
                'total_trades': metrics.total_trades,
                'total_return': metrics.total_return,
                'annual_return': metrics.annual_return,
                'max_drawdown': metrics.max_drawdown
            },
            'metrics': asdict(metrics),
            'recent_trades': [asdict(trade) for trade in self.get_recent_trades(10)],
            'equity_curve_length': len(self.equity_curve),
            'update_timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance breakdown by strategy"""
        with self.lock:
            strategy_stats = defaultdict(lambda: {
                'trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'total_volume': 0.0
            })
            
            for trade in self.trades:
                stats = strategy_stats[trade.strategy]
                stats['trades'] += 1
                stats['total_pnl'] += trade.pnl
                stats['total_volume'] += trade.quantity * trade.entry_price
                
                if trade.pnl > 0:
                    stats['winning_trades'] += 1
            
            # Calculate additional metrics
            for strategy, stats in strategy_stats.items():
                stats['win_rate'] = stats['winning_trades'] / stats['trades'] if stats['trades'] > 0 else 0
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            return dict(strategy_stats)

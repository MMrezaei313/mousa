from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self, name: str = "BaseStrategy", **params):
        self.name = name
        self.params = params
        self.signals = pd.DataFrame()
        self.positions = pd.Series(dtype=float)
        self.returns = pd.Series(dtype=float)
        self.logger = self._setup_logging()
        self.is_initialized = False
        
    def _setup_logging(self):
        """Setup strategy-specific logging"""
        logger = logging.getLogger(f"{__name__}.{self.name}")
        return logger
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data
        Returns DataFrame with signals and additional info
        """
        pass
    
    def calculate_returns(self, data: pd.DataFrame, signals: pd.DataFrame = None) -> pd.Series:
        """Calculate strategy returns based on signals"""
        if signals is None:
            signals = self.signals
            
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' prices")
            
        price_returns = data['close'].pct_change()
        
        if 'position' not in signals.columns:
            raise ValueError("Signals must contain 'position' column")
            
        self.returns = signals['position'].shift(1) * price_returns
        self.returns.name = f'{self.name}_returns'
        
        return self.returns
    
    def calculate_performance_metrics(self, returns: pd.Series = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if returns is None:
            returns = self.returns
            
        if returns.empty:
            return {}
            
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) < 2:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate and profit factor
        winning_trades = len(returns[returns > 0])
        losing_trades = len(returns[returns < 0])
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
            'avg_trade_return': returns.mean(),
            'best_trade': returns.max(),
            'worst_trade': returns.min()
        }
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, Any]:
        """Run complete backtest for the strategy"""
        self.logger.info(f"Running backtest for {self.name}")
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Calculate returns
        returns = self.calculate_returns(data, signals)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(returns)
        
        # Calculate equity curve
        equity_curve = initial_capital * (1 + returns).cumprod()
        
        # Trade analysis
        trades = self.analyze_trades(signals, data)
        
        return {
            'strategy_name': self.name,
            'parameters': self.params,
            'signals': signals,
            'returns': returns,
            'equity_curve': equity_curve,
            'performance_metrics': metrics,
            'trades': trades,
            'data_period': {
                'start': data.index.min(),
                'end': data.index.max(),
                'total_periods': len(data)
            }
        }
    
    def analyze_trades(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze individual trades"""
        if 'position' not in signals.columns:
            return pd.DataFrame()
            
        position_changes = signals['position'].diff().fillna(0)
        trade_entries = position_changes != 0
        
        trades = []
        current_trade = None
        
        for i, (timestamp, row) in enumerate(signals.iterrows()):
            if trade_entries.iloc[i]:
                if current_trade is not None:
                    # Close previous trade
                    current_trade['exit_price'] = data['close'].iloc[i]
                    current_trade['exit_time'] = timestamp
                    current_trade['pnl'] = (
                        current_trade['exit_price'] - current_trade['entry_price']
                    ) * current_trade['position']
                    current_trade['pnl_pct'] = (
                        current_trade['pnl'] / 
                        (current_trade['entry_price'] * abs(current_trade['position']))
                    )
                    trades.append(current_trade)
                
                # Start new trade
                current_trade = {
                    'entry_time': timestamp,
                    'entry_price': data['close'].iloc[i],
                    'position': row['position'],
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': 0,
                    'pnl_pct': 0
                }
        
        return pd.DataFrame(trades)
    
    def optimize_parameters(self, data: pd.DataFrame, param_space: Dict, 
                          objective: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimize strategy parameters"""
        from ..optimization.genetic_optimizer import GeneticOptimizer
        
        def objective_function(params, data):
            try:
                self.params.update(params)
                signals = self.generate_signals(data)
                returns = self.calculate_returns(data, signals)
                metrics = self.calculate_performance_metrics(returns)
                return metrics.get(objective, -np.inf)
            except Exception as e:
                self.logger.warning(f"Parameter optimization failed: {e}")
                return -np.inf
        
        optimizer = GeneticOptimizer(objective_function=objective_function)
        result = optimizer.optimize(param_space, data)
        
        return result
    
    def __str__(self):
        return f"{self.name}({self.params})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"

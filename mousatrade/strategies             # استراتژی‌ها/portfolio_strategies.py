import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from ..optimization.portfolio_optimizer import PortfolioOptimizer

from .base_strategy import BaseStrategy

class PortfolioStrategy(BaseStrategy):
    """
    Portfolio-based strategy that combines multiple sub-strategies
    """
    
    def __init__(self, strategies: Dict[str, BaseStrategy], 
                 allocation_method: str = 'equal', **kwargs):
        params = {
            'strategies': strategies,
            'allocation_method': allocation_method
        }
        params.update(kwargs)
        super().__init__("PortfolioStrategy", **params)
        
        self.strategies = strategies
        self.weights = {}
        self.strategy_returns = {}
        
    def calculate_strategy_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns for all sub-strategies"""
        strategy_returns = {}
        
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                returns = strategy.calculate_returns(data, signals)
                strategy_returns[name] = returns
            except Exception as e:
                self.logger.warning(f"Strategy {name} failed: {e}")
                continue
        
        self.strategy_returns = strategy_returns
        return pd.DataFrame(strategy_returns).dropna()
    
    def optimize_allocation(self, data: pd.DataFrame, 
                          method: str = 'mean_variance') -> Dict[str, float]:
        """Optimize allocation weights across strategies"""
        strategy_returns_df = self.calculate_strategy_returns(data)
        
        if strategy_returns_df.empty:
            self.logger.warning("No valid strategy returns for optimization")
            return self._equal_weights()
        
        # Use portfolio optimizer
        optimizer = PortfolioOptimizer()
        
        if method == 'mean_variance':
            result = optimizer.mean_variance_optimization(strategy_returns_df)
        elif method == 'risk_parity':
            result = optimizer.risk_parity_optimization(strategy_returns_df)
        elif method == 'equal':
            return self._equal_weights()
        else:
            self.logger.warning(f"Unknown allocation method: {method}")
            return self._equal_weights()
        
        if result['success']:
            self.weights = result['weights'].to_dict()
            return self.weights
        else:
            self.logger.warning("Portfolio optimization failed, using equal weights")
            return self._equal_weights()
    
    def _equal_weights(self) -> Dict[str, float]:
        """Calculate equal weights for all strategies"""
        n_strategies = len(self.strategies)
        if n_strategies == 0:
            return {}
        
        equal_weight = 1.0 / n_strategies
        return {name: equal_weight for name in self.strategies.keys()}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate portfolio signals by combining sub-strategy signals"""
        if not self.weights:
            self.weights = self.optimize_allocation(data, self.params['allocation_method'])
        
        # Generate signals from all strategies
        all_signals = {}
        valid_strategies = []
        
        for name, strategy in self.strategies.items():
            if name not in self.weights:
                continue
                
            try:
                signals = strategy.generate_signals(data)
                if 'position' in signals.columns:
                    all_signals[name] = signals['position']
                    valid_strategies.append(name)
            except Exception as e:
                self.logger.warning(f"Strategy {name} signal generation failed: {e}")
                continue
        
        if not all_signals:
            raise ValueError("No valid strategies available")
        
        # Combine signals using optimized weights
        signals_df = pd.DataFrame(all_signals)
        
        # Calculate weighted position
        weights_series = pd.Series(self.weights)
        valid_weights = weights_series[valid_strategies]
        
        # Normalize weights for available strategies
        valid_weights = valid_weights / valid_weights.sum()
        
        portfolio_signals = pd.DataFrame(index=data.index)
        portfolio_signals['position'] = signals_df.dot(valid_weights)
        
        # Add strategy contributions
        for strategy in valid_strategies:
            portfolio_signals[f'{strategy}_position'] = signals_df[strategy]
            portfolio_signals[f'{strategy}_weight'] = valid_weights[strategy]
        
        # Add portfolio metrics
        portfolio_signals['strategy_count'] = len(valid_strategies)
        portfolio_signals['weight_sum'] = valid_weights.sum()
        
        self.signals = portfolio_signals
        return portfolio_signals
    
    def get_portfolio_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze portfolio composition and performance"""
        if not self.strategy_returns:
            self.calculate_strategy_returns(data)
        
        analysis = {
            'strategy_weights': self.weights,
            'strategy_count': len(self.strategies),
            'active_strategies': list(self.weights.keys()),
            'allocation_method': self.params['allocation_method']
        }
        
        # Calculate individual strategy performance
        strategy_performance = {}
        for name, returns in self.strategy_returns.items():
            if returns is not None and len(returns) > 0:
                metrics = self.calculate_performance_metrics(returns)
                strategy_performance[name] = metrics
        
        analysis['strategy_performance'] = strategy_performance
        
        return analysis

class RiskParityStrategy(PortfolioStrategy):
    """
    Risk Parity based portfolio strategy
    """
    
    def __init__(self, strategies: Dict[str, BaseStrategy], **kwargs):
        super().__init__(strategies, allocation_method='risk_parity', **kwargs)
        self.name = "RiskParityStrategy"

class MomentumPortfolioStrategy(PortfolioStrategy):
    """
    Momentum-based portfolio strategy
    """
    
    def __init__(self, strategies: Dict[str, BaseStrategy], 
                 lookback_period: int = 63, **kwargs):
        params = {
            'lookback_period': lookback_period
        }
        params.update(kwargs)
        super().__init__(strategies, allocation_method='momentum', **params)
        self.name = "MomentumPortfolioStrategy"
    
    def optimize_allocation(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Momentum-based allocation"""
        strategy_returns_df = self.calculate_strategy_returns(data)
        
        if strategy_returns_df.empty:
            return self._equal_weights()
        
        # Calculate momentum (recent performance)
        lookback = min(self.params['lookback_period'], len(strategy_returns_df))
        recent_returns = strategy_returns_df.tail(lookback)
        momentum = (1 + recent_returns).prod() - 1
        
        # Use momentum as weights (positive momentum only)
        positive_momentum = momentum[momentum > 0]
        
        if len(positive_momentum) == 0:
            return self._equal_weights()
        
        # Normalize weights
        weights = positive_momentum / positive_momentum.sum()
        self.weights = weights.to_dict()
        
        return self.weights

# Portfolio strategy factory
def create_portfolio_strategy(strategy_type: str = 'basic', 
                            strategies: Dict[str, BaseStrategy] = None,
                            **params) -> PortfolioStrategy:
    """Factory function for portfolio strategies"""
    portfolio_strategies = {
        'basic': PortfolioStrategy,
        'risk_parity': RiskParityStrategy,
        'momentum': MomentumPortfolioStrategy
    }
    
    if strategy_type not in portfolio_strategies:
        raise ValueError(f"Unknown portfolio strategy: {strategy_type}")
    
    return portfolio_strategies[strategy_type](strategies or {}, **params)

"""
Tests for trading strategies module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from mousatrade.strategies import (
    BaseStrategy,
    MovingAverageCrossover,
    RSIMeanReversion, 
    MACDStrategy,
    BollingerBandsStrategy,
    RandomForestStrategy,
    PortfolioStrategy,
    create_strategy,
    create_ml_strategy,
    create_portfolio_strategy
)


class TestBaseStrategy:
    """Test cases for BaseStrategy class"""
    
    def test_base_strategy_initialization(self):
        """Test BaseStrategy initialization"""
        strategy = BaseStrategy(name="TestStrategy", param1=10, param2=20)
        
        assert strategy.name == "TestStrategy"
        assert strategy.params == {"param1": 10, "param2": 20}
        assert strategy.signals.empty
        assert strategy.positions.empty
        assert strategy.returns.empty
        assert not strategy.is_initialized
    
    def test_base_strategy_abstract_method(self, sample_market_data):
        """Test that BaseStrategy cannot be instantiated directly"""
        strategy = BaseStrategy(name="TestStrategy")
        
        with pytest.raises(NotImplementedError):
            strategy.generate_signals(sample_market_data)
    
    def test_calculate_returns(self, sample_market_data):
        """Test returns calculation"""
        class ConcreteStrategy(BaseStrategy):
            def generate_signals(self, data):
                signals = pd.DataFrame(index=data.index)
                signals['position'] = 1  # Always long
                return signals
        
        strategy = ConcreteStrategy(name="TestStrategy")
        signals = strategy.generate_signals(sample_market_data)
        returns = strategy.calculate_returns(sample_market_data, signals)
        
        assert len(returns) > 0
        assert returns.name == "TestStrategy_returns"
        assert not returns.isna().any()
    
    def test_performance_metrics(self, sample_market_data):
        """Test performance metrics calculation"""
        class ConcreteStrategy(BaseStrategy):
            def generate_signals(self, data):
                signals = pd.DataFrame(index=data.index)
                signals['position'] = np.random.choice([-1, 0, 1], len(data))
                return signals
        
        strategy = ConcreteStrategy(name="TestStrategy")
        signals = strategy.generate_signals(sample_market_data)
        returns = strategy.calculate_returns(sample_market_data, signals)
        metrics = strategy.calculate_performance_metrics(returns)
        
        expected_metrics = [
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'profit_factor', 'calmar_ratio',
            'total_trades', 'avg_trade_return', 'best_trade', 'worst_trade'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics


class TestTechnicalStrategies:
    """Test cases for technical trading strategies"""
    
    def test_moving_average_crossover(self, sample_market_data):
        """Test Moving Average Crossover strategy"""
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
        signals = strategy.generate_signals(sample_market_data)
        
        assert not signals.empty
        assert 'fast_ma' in signals.columns
        assert 'slow_ma' in signals.columns
        assert 'position' in signals.columns
        assert 'crossover' in signals.columns
        
        # Test signal generation
        assert signals['position'].isin([-1, 1]).all()
    
    def test_rsi_mean_reversion(self, sample_market_data):
        """Test RSI Mean Reversion strategy"""
        strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(sample_market_data)
        
        assert not signals.empty
        assert 'rsi' in signals.columns
        assert 'position' in signals.columns
        assert 'signal_strength' in signals.columns
        
        # Test RSI calculation
        assert signals['rsi'].between(0, 100).all()
        
        # Test signal bounds
        assert signals['position'].isin([-1, 0, 1]).all()
    
    def test_macd_strategy(self, sample_market_data):
        """Test MACD strategy"""
        strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        signals = strategy.generate_signals(sample_market_data)
        
        assert not signals.empty
        assert 'macd' in signals.columns
        assert 'signal_line' in signals.columns
        assert 'histogram' in signals.columns
        assert 'position' in signals.columns
        
        # Test MACD calculation
        assert not signals['macd'].isna().all()
    
    def test_bollinger_bands_strategy(self, sample_market_data):
        """Test Bollinger Bands strategy"""
        strategy = BollingerBandsStrategy(period=20, num_std=2.0)
        signals = strategy.generate_signals(sample_market_data)
        
        assert not signals.empty
        assert 'middle_band' in signals.columns
        assert 'upper_band' in signals.columns
        assert 'lower_band' in signals.columns
        assert 'band_position' in signals.columns
        assert 'position' in signals.columns
        
        # Test band position calculation
        assert signals['band_position'].between(0, 1).all()


class TestMLStrategies:
    """Test cases for machine learning strategies"""
    
    @pytest.mark.slow
    def test_random_forest_strategy(self, sample_market_data, sample_ml_features):
        """Test Random Forest strategy"""
        strategy = RandomForestStrategy(n_estimators=10, max_depth=5)  # Small for testing
        
        # Test feature creation
        features = strategy.create_features(sample_market_data)
        assert not features.empty
        assert len(strategy.feature_columns) > 0
        
        # Test target creation
        target = strategy.create_target(sample_market_data)
        assert not target.empty
        assert target.isin([0, 1]).all()
        
        # Test training
        training_result = strategy.train(sample_market_data)
        assert strategy.is_trained
        assert 'train_accuracy' in training_result
        assert 'test_accuracy' in training_result
        
        # Test signal generation
        signals = strategy.generate_signals(sample_market_data)
        assert not signals.empty
        assert 'prediction' in signals.columns
        assert 'probability' in signals.columns
        assert 'position' in signals.columns
    
    def test_ml_strategy_factory(self):
        """Test ML strategy factory function"""
        strategy = create_ml_strategy('random_forest', n_estimators=10)
        assert isinstance(strategy, RandomForestStrategy)
        assert strategy.params['n_estimators'] == 10


class TestPortfolioStrategies:
    """Test cases for portfolio strategies"""
    
    def test_portfolio_strategy_initialization(self):
        """Test PortfolioStrategy initialization"""
        # Create some mock strategies
        strategies = {
            'ma': MovingAverageCrossover(fast_window=10, slow_window=30),
            'rsi': RSIMeanReversion(rsi_period=14)
        }
        
        portfolio = PortfolioStrategy(strategies, allocation_method='equal')
        
        assert portfolio.name == "PortfolioStrategy"
        assert len(portfolio.strategies) == 2
        assert portfolio.params['allocation_method'] == 'equal'
    
    def test_portfolio_strategy_allocation(self, sample_market_data):
        """Test portfolio allocation calculation"""
        strategies = {
            'ma': MovingAverageCrossover(fast_window=10, slow_window=30),
            'rsi': RSIMeanReversion(rsi_period=14)
        }
        
        portfolio = PortfolioStrategy(strategies, allocation_method='equal')
        
        # Test strategy returns calculation
        strategy_returns = portfolio.calculate_strategy_returns(sample_market_data)
        assert not strategy_returns.empty
        assert len(strategy_returns.columns) == len(strategies)
        
        # Test allocation optimization
        weights = portfolio.optimize_allocation(sample_market_data)
        assert len(weights) == len(strategies)
        assert abs(sum(weights.values()) - 1.0) < 1e-10  # Weights sum to 1
    
    def test_portfolio_signal_generation(self, sample_market_data):
        """Test portfolio signal generation"""
        strategies = {
            'ma': MovingAverageCrossover(fast_window=10, slow_window=30),
            'rsi': RSIMeanReversion(rsi_period=14)
        }
        
        portfolio = PortfolioStrategy(strategies)
        signals = portfolio.generate_signals(sample_market_data)
        
        assert not signals.empty
        assert 'position' in signals.columns
        
        # Should have contributions from each strategy
        for strategy_name in strategies.keys():
            assert f'{strategy_name}_position' in signals.columns
            assert f'{strategy_name}_weight' in signals.columns


class TestStrategyFactories:
    """Test cases for strategy factory functions"""
    
    def test_create_strategy_technical(self):
        """Test technical strategy factory"""
        strategy = create_strategy('moving_average_crossover', 
                                 fast_window=10, slow_window=30)
        assert isinstance(strategy, MovingAverageCrossover)
        assert strategy.params['fast_window'] == 10
        
        strategy = create_strategy('rsi_mean_reversion',
                                 rsi_period=14, oversold=30, overbought=70)
        assert isinstance(strategy, RSIMeanReversion)
    
    def test_create_strategy_invalid(self):
        """Test factory with invalid strategy name"""
        with pytest.raises(ValueError):
            create_strategy('invalid_strategy')
    
    def test_create_portfolio_strategy(self):
        """Test portfolio strategy factory"""
        strategies = {
            'ma': MovingAverageCrossover(fast_window=10, slow_window=30)
        }
        
        portfolio = create_portfolio_strategy('basic', strategies)
        assert isinstance(portfolio, PortfolioStrategy)


class TestStrategyBacktesting:
    """Test cases for strategy backtesting"""
    
    def test_strategy_backtest(self, sample_market_data):
        """Test complete strategy backtest"""
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
        backtest_results = strategy.backtest(sample_market_data, initial_capital=10000)
        
        expected_keys = [
            'strategy_name', 'parameters', 'signals', 'returns',
            'equity_curve', 'performance_metrics', 'trades', 'data_period'
        ]
        
        for key in expected_keys:
            assert key in backtest_results
        
        # Test performance metrics
        metrics = backtest_results['performance_metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # Test equity curve
        equity_curve = backtest_results['equity_curve']
        assert len(equity_curve) > 0
        assert equity_curve.iloc[0] == 10000  # Initial capital
    
    @pytest.mark.slow
    def test_strategy_optimization(self, sample_market_data):
        """Test strategy parameter optimization"""
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
        
        param_space = {
            'fast_window': (5, 20, 'int'),
            'slow_window': (25, 50, 'int')
        }
        
        optimization_result = strategy.optimize_parameters(
            sample_market_data, param_space, objective='sharpe_ratio'
        )
        
        assert 'best_parameters' in optimization_result
        assert 'best_fitness' in optimization_result
        assert 'fitness_history' in optimization_result
        
        best_params = optimization_result['best_parameters']
        assert 'fast_window' in best_params
        assert 'slow_window' in best_params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

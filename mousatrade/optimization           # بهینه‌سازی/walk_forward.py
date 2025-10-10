import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional
from datetime import datetime, timedelta
import logging
from .genetic_optimizer import GeneticOptimizer
from .bayesian_optimizer import BayesianOptimizer

class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for robust trading strategy validation
    Implements rolling window optimization to avoid overfitting
    """
    
    def __init__(self,
                 window_size: int = 252,
                 step_size: int = 63,
                 optimization_method: str = 'genetic',
                 metric: str = 'sharpe_ratio',
                 min_periods: int = 50):
        
        self.window_size = window_size  # Training window size (days)
        self.step_size = step_size      # Step size between windows (days)
        self.optimization_method = optimization_method
        self.metric = metric
        self.min_periods = min_periods
        
        self.optimization_results = []
        self.out_of_sample_results = []
        self.parameter_evolution = []
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for walk-forward optimizer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _create_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Create training and testing windows"""
        windows = []
        dates = data.index
        
        start_idx = 0
        end_idx = start_idx + self.window_size
        
        while end_idx + self.step_size <= len(dates):
            train_start = dates[start_idx]
            train_end = dates[end_idx - 1]
            test_start = dates[end_idx]
            test_end = dates[min(end_idx + self.step_size - 1, len(dates) - 1)]
            
            windows.append(((train_start, train_end), (test_start, test_end)))
            
            start_idx += self.step_size
            end_idx = start_idx + self.window_size
        
        self.logger.info(f"Created {len(windows)} walk-forward windows")
        return windows
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(returns) < self.min_periods:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
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
        avg_win = returns[returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = returns[returns < 0].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else np.inf
        
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
            'total_trades': total_trades
        }
    
    def _get_optimizer(self, parameter_space: Dict) -> Any:
        """Get appropriate optimizer based on method"""
        if self.optimization_method == 'genetic':
            return GeneticOptimizer(
                population_size=30,
                generations=50,
                objective_function=self.objective_function
            )
        elif self.optimization_method == 'bayesian':
            return BayesianOptimizer(
                n_iter=50,
                init_points=10,
                objective_function=self.objective_function
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def walk_forward_optimize(self,
                            data: pd.DataFrame,
                            parameter_space: Dict[str, Tuple],
                            strategy_function: Callable,
                            objective_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Perform walk-forward optimization
        
        Args:
            data: Market data DataFrame
            parameter_space: Parameter space for optimization
            strategy_function: Function that generates trades given parameters and data
            objective_function: Custom objective function (optional)
        """
        
        self.logger.info("Starting walk-forward optimization...")
        self.strategy_function = strategy_function
        self.objective_function = objective_function or self._default_objective
        
        # Create windows
        windows = self._create_windows(data)
        
        for i, ((train_start, train_end), (test_start, test_end)) in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Split data
            train_data = data.loc[train_start:train_end].copy()
            test_data = data.loc[test_start:test_end].copy()
            
            if len(train_data) < self.min_periods or len(test_data) < self.min_periods:
                self.logger.warning(f"Window {i+1} has insufficient data, skipping")
                continue
            
            # Optimize on training data
            optimizer = self._get_optimizer(parameter_space)
            optimization_result = optimizer.optimize(parameter_space, train_data)
            
            # Test on out-of-sample data
            best_params = optimization_result['best_parameters']
            oos_performance = self._test_parameters(best_params, test_data)
            
            # Store results
            self.optimization_results.append({
                'window': i + 1,
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end),
                'parameters': best_params,
                'train_performance': optimization_result['best_fitness'],
                'test_performance': oos_performance
            })
            
            self.parameter_evolution.append({
                'window': i + 1,
                'period_end': train_end,
                'parameters': best_params
            })
        
        return self._generate_final_report(data)
    
    def _default_objective(self, params: Dict, data: pd.DataFrame) -> float:
        """Default objective function using specified metric"""
        try:
            # Generate strategy returns
            returns = self.strategy_function(params, data)
            
            if returns is None or len(returns) < self.min_periods:
                return -np.inf
            
            # Calculate specified metric
            if self.metric == 'sharpe_ratio':
                annual_return = returns.mean() * 252
                volatility = returns.std() * np.sqrt(252)
                return annual_return / volatility if volatility > 0 else -np.inf
            
            elif self.metric == 'total_return':
                return (1 + returns).prod() - 1
            
            elif self.metric == 'calmar_ratio':
                total_return = (1 + returns).prod() - 1
                annual_return = (1 + total_return) ** (252 / len(returns)) - 1
                cumulative_returns = (1 + returns).cumprod()
                max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
                return annual_return / abs(max_drawdown) if max_drawdown != 0 else -np.inf
            
            elif self.metric == 'profit_factor':
                winning_returns = returns[returns > 0]
                losing_returns = returns[returns < 0]
                gross_profit = winning_returns.sum() if len(winning_returns) > 0 else 0
                gross_loss = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
                return gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
                
        except Exception as e:
            self.logger.warning(f"Objective function failed: {e}")
            return -np.inf
    
    def _test_parameters(self, params: Dict, test_data: pd.DataFrame) -> Dict[str, float]:
        """Test parameters on out-of-sample data"""
        try:
            returns = self.strategy_function(params, test_data)
            
            if returns is None or len(returns) < self.min_periods:
                return {}
            
            return self._calculate_performance_metrics(returns)
            
        except Exception as e:
            self.logger.warning(f"Parameter testing failed: {e}")
            return {}
    
    def _generate_final_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive walk-forward analysis report"""
        
        # Calculate overall performance
        all_returns = []
        for result in self.optimization_results:
            if 'test_performance' in result and 'total_return' in result['test_performance']:
                # Simulate cumulative returns (simplified)
                test_period_returns = result['test_performance'].get('annual_return', 0) / 252
                period_days = (result['test_period'][1] - result['test_period'][0]).days
                period_returns = [test_period_returns] * max(1, period_days)
                all_returns.extend(period_returns)
        
        overall_metrics = self._calculate_performance_metrics(pd.Series(all_returns)) if all_returns else {}
        
        # Parameter stability analysis
        param_stability = self._analyze_parameter_stability()
        
        # Overfitting analysis
        overfitting_analysis = self._analyze_overfitting()
        
        return {
            'overall_performance': overall_metrics,
            'parameter_stability': param_stability,
            'overfitting_analysis': overfitting_analysis,
            'window_results': self.optimization_results,
            'parameter_evolution': self.parameter_evolution,
            'summary_statistics': self._calculate_summary_statistics()
        }
    
    def _analyze_parameter_stability(self) -> Dict[str, Any]:
        """Analyze stability of optimized parameters across windows"""
        if not self.parameter_evolution:
            return {}
        
        # Extract parameter values
        param_values = {}
        for evolution in self.parameter_evolution:
            for param, value in evolution['parameters'].items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)
        
        # Calculate stability metrics
        stability_metrics = {}
        for param, values in param_values.items():
            if len(values) > 1:
                stability_metrics[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf,
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': self._calculate_trend(values)
                }
        
        return stability_metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend of parameter values"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def _analyze_overfitting(self) -> Dict[str, Any]:
        """Analyze overfitting by comparing in-sample vs out-of-sample performance"""
        if not self.optimization_results:
            return {}
        
        train_scores = [r['train_performance'] for r in self.optimization_results]
        test_scores = [r['test_performance'].get(self.metric, 0) for r in self.optimization_results]
        
        # Calculate overfitting metrics
        correlation = np.corrcoef(train_scores, test_scores)[0, 1] if len(train_scores) > 1 else 0
        performance_decay = np.mean(test_scores) / np.mean(train_scores) if np.mean(train_scores) != 0 else 0
        
        return {
            'train_test_correlation': correlation,
            'performance_decay_ratio': performance_decay,
            'avg_train_score': np.mean(train_scores),
            'avg_test_score': np.mean(test_scores),
            'overfitting_degree': 1 - performance_decay
        }
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for the walk-forward optimization"""
        if not self.optimization_results:
            return {}
        
        successful_windows = len(self.optimization_results)
        total_windows = len(self.parameter_evolution)
        
        test_performances = [r['test_performance'].get(self.metric, 0) 
                           for r in self.optimization_results 
                           if 'test_performance' in r]
        
        return {
            'total_windows': total_windows,
            'successful_windows': successful_windows,
            'success_rate': successful_windows / total_windows if total_windows > 0 else 0,
            'avg_test_performance': np.mean(test_performances) if test_performances else 0,
            'std_test_performance': np.std(test_performances) if test_performances else 0,
            'best_window_performance': np.max(test_performances) if test_performances else 0,
            'worst_window_performance': np.min(test_performances) if test_performances else 0
        }
    
    def get_robust_parameters(self, method: str = 'mean') -> Dict[str, float]:
        """Get robust parameters based on walk-forward analysis"""
        if not self.parameter_evolution:
            return {}
        
        if method == 'mean':
            return self._get_mean_parameters()
        elif method == 'median':
            return self._get_median_parameters()
        elif method == 'best':
            return self._get_best_performing_parameters()
        elif method == 'most_stable':
            return self._get_most_stable_parameters()
        else:
            raise ValueError(f"Unknown robust parameter method: {method}")
    
    def _get_mean_parameters(self) -> Dict[str, float]:
        """Get mean parameter values across all windows"""
        param_sums = {}
        param_counts = {}
        
        for evolution in self.parameter_evolution:
            for param, value in evolution['parameters'].items():
                if param not in param_sums:
                    param_sums[param] = 0
                    param_counts[param] = 0
                
                param_sums[param] += value
                param_counts[param] += 1
        
        return {param: param_sums[param] / param_counts[param] for param in param_sums}
    
    def _get_median_parameters(self) -> Dict[str, float]:
        """Get median parameter values across all windows"""
        param_values = {}
        
        for evolution in self.parameter_evolution:
            for param, value in evolution['parameters'].items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)
        
        return {param: np.median(values) for param, values in param_values.items()}
    
    def _get_best_performing_parameters(self) -> Dict[str, float]:
        """Get parameters from the best performing window"""
        if not self.optimization_results:
            return {}
        
        best_idx = np.argmax([r['test_performance'].get(self.metric, -np.inf) 
                            for r in self.optimization_results])
        
        return self.optimization_results[best_idx]['parameters']
    
    def _get_most_stable_parameters(self) -> Dict[str, float]:
        """Get parameters with lowest variability across windows"""
        stability_metrics = self._analyze_parameter_stability()
        
        if not stability_metrics:
            return {}
        
        # Select parameters with lowest coefficient of variation
        stable_params = {}
        for param, metrics in stability_metrics.items():
            if metrics['cv'] < 0.5:  # Threshold for stability
                stable_params[param] = metrics['mean']
        
        return stable_params

# Example strategy function for testing
def sample_moving_average_strategy(params: Dict, data: pd.DataFrame) -> pd.Series:
    """Sample moving average crossover strategy"""
    try:
        fast_ma = data['close'].rolling(window=params['fast_window']).mean()
        slow_ma = data['close'].rolling(window=params['slow_window']).mean()
        
        # Generate signals
        signals = np.where(fast_ma > slow_ma, 1, -1)
        positions = pd.Series(signals, index=data.index).shift(1)
        
        # Calculate returns
        returns = data['close'].pct_change()
        strategy_returns = positions * returns
        
        return strategy_returns.dropna()
        
    except Exception as e:
        print(f"Strategy error: {e}")
        return pd.Series()

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 10, 1000),
        'high': np.random.normal(105, 10, 1000),
        'low': np.random.normal(95, 10, 1000),
        'close': 100 + np.cumsum(np.random.normal(0, 1, 1000)),
        'volume': np.random.normal(1000, 200, 1000)
    }, index=dates)
    
    # Define parameter space
    parameter_space = {
        'fast_window': (5, 50, 'int'),
        'slow_window': (20, 200, 'int')
    }
    
    # Perform walk-forward optimization
    wf_optimizer = WalkForwardOptimizer(
        window_size=200,
        step_size=50,
        optimization_method='genetic',
        metric='sharpe_ratio'
    )
    
    results = wf_optimizer.walk_forward_optimize(
        data=sample_data,
        parameter_space=parameter_space,
        strategy_function=sample_moving_average_strategy
    )
    
    print("Walk-Forward Optimization Results:")
    print(f"Overall Performance: {results['overall_performance']}")
    print(f"Parameter Stability: {results['parameter_stability']}")
    print(f"Overfitting Analysis: {results['overfitting_analysis']}")
    
    # Get robust parameters
    robust_params = wf_optimizer.get_robust_parameters(method='mean')
    print(f"Robust Parameters (mean): {robust_params}")

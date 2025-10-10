import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize
from scipy import linalg
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization for Multiple Trading Strategies
    Implements Modern Portfolio Theory and Risk-Based Allocation
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 target_return: float = None,
                 risk_aversion: float = 1.0,
                 max_allocation: float = 0.3,
                 min_allocation: float = 0.0):
        
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.risk_aversion = risk_aversion
        self.max_allocation = max_allocation
        self.min_allocation = min_allocation
        
        self.weights = None
        self.performance_metrics = {}
        self.optimization_history = []
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for portfolio optimizer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def mean_variance_optimization(self, 
                                 returns: pd.DataFrame,
                                 method: str = 'sharpe') -> Dict[str, Any]:
        """
        Mean-Variance Optimization (Markowitz)
        
        Args:
            returns: DataFrame of strategy returns (each column is a strategy)
            method: Optimization objective ('sharpe', 'min_volatility', 'max_return')
        """
        
        self.logger.info("Starting Mean-Variance Optimization...")
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Number of assets
        n_assets = len(expected_returns)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds for weights
        bounds = tuple((self.min_allocation, self.max_allocation) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Define objective function based on method
        if method == 'sharpe':
            # Maximize Sharpe Ratio
            objective = lambda w: -self._sharpe_ratio(w, expected_returns, cov_matrix)
        elif method == 'min_volatility':
            # Minimize Volatility
            objective = lambda w: self._portfolio_volatility(w, cov_matrix)
        elif method == 'max_return':
            # Maximize Return for given volatility
            target_volatility = returns.std().mean()
            objective = lambda w: -self._portfolio_return(w, expected_returns) + \
                                 self.risk_aversion * abs(self._portfolio_volatility(w, cov_matrix) - target_volatility)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Perform optimization
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if result.success:
            self.weights = pd.Series(result.x, index=returns.columns)
            self._calculate_performance_metrics(returns)
            
            self.logger.info("Mean-Variance Optimization completed successfully")
            return {
                'weights': self.weights,
                'performance': self.performance_metrics,
                'optimization_method': f'mean_variance_{method}',
                'success': True
            }
        else:
            self.logger.error(f"Mean-Variance Optimization failed: {result.message}")
            return {'success': False, 'message': result.message}
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Risk Parity Optimization (Equal Risk Contribution)
        """
        
        self.logger.info("Starting Risk Parity Optimization...")
        
        cov_matrix = returns.cov()
        n_assets = len(cov_matrix)
        
        def risk_parity_objective(weights):
            """Objective function for risk parity"""
            portfolio_volatility = self._portfolio_volatility(weights, cov_matrix)
            risk_contributions = self._risk_contribution(weights, cov_matrix)
            
            # Target: equal risk contribution (1/n for each asset)
            target_risk = np.array([1.0 / n_assets] * n_assets)
            
            # Sum of squared differences from equal risk contribution
            error = np.sum((risk_contributions - target_risk) ** 2)
            return error
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((self.min_allocation, self.max_allocation) for _ in range(n_assets))
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if result.success:
            self.weights = pd.Series(result.x, index=returns.columns)
            self._calculate_performance_metrics(returns)
            
            self.logger.info("Risk Parity Optimization completed successfully")
            return {
                'weights': self.weights,
                'performance': self.performance_metrics,
                'optimization_method': 'risk_parity',
                'success': True
            }
        else:
            self.logger.error(f"Risk Parity Optimization failed: {result.message}")
            return {'success': False, 'message': result.message}
    
    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Hierarchical Risk Parity (HRP) - Advanced portfolio construction
        """
        
        self.logger.info("Starting Hierarchical Risk Parity Optimization...")
        
        try:
            # Step 1: Hierarchical clustering of assets
            correlation_matrix = returns.corr()
            distance_matrix = np.sqrt((1 - correlation_matrix) / 2)
            
            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
            linkage_matrix = linkage(distance_matrix, method='ward')
            leaf_order = leaves_list(linkage_matrix)
            
            # Step 2: Quasi-diagonalization
            ordered_returns = returns.iloc[:, leaf_order]
            ordered_corr = correlation_matrix.iloc[leaf_order, leaf_order]
            
            # Step 3: Recursive bisection
            weights = self._hrp_recursive_bisection(ordered_returns, ordered_corr)
            
            # Reorder weights to original asset order
            original_order = [returns.columns.get_loc(returns.columns[leaf_order[i]]) 
                            for i in range(len(leaf_order))]
            reordered_weights = np.zeros_like(weights)
            reordered_weights[original_order] = weights
            
            self.weights = pd.Series(reordered_weights, index=returns.columns)
            self._calculate_performance_metrics(returns)
            
            self.logger.info("Hierarchical Risk Parity completed successfully")
            return {
                'weights': self.weights,
                'performance': self.performance_metrics,
                'optimization_method': 'hierarchical_risk_parity',
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"HRP Optimization failed: {e}")
            return {'success': False, 'message': str(e)}
    
    def _hrp_recursive_bisection(self, returns: pd.DataFrame, 
                               correlation: pd.DataFrame) -> np.ndarray:
        """Recursive bisection for HRP"""
        n_assets = len(returns.columns)
        
        if n_assets == 1:
            return np.array([1.0])
        
        # Split into two clusters
        split_point = n_assets // 2
        cluster1 = returns.columns[:split_point]
        cluster2 = returns.columns[split_point:]
        
        # Recursively get weights for each cluster
        weights1 = self._hrp_recursive_bisection(returns[cluster1], 
                                               correlation.loc[cluster1, cluster1])
        weights2 = self._hrp_recursive_bisection(returns[cluster2], 
                                               correlation.loc[cluster2, cluster2])
        
        # Calculate cluster variances
        var1 = self._portfolio_volatility(weights1, returns[cluster1].cov())
        var2 = self._portfolio_volatility(weights2, returns[cluster2].cov())
        
        # Allocate based on inverse volatility
        alpha = 1 - var1 / (var1 + var2)
        
        # Combine weights
        combined_weights = np.zeros(n_assets)
        combined_weights[:split_point] = alpha * weights1
        combined_weights[split_point:] = (1 - alpha) * weights2
        
        return combined_weights
    
    def black_litterman_optimization(self,
                                   returns: pd.DataFrame,
                                   market_caps: pd.Series = None,
                                   views: Dict[str, float] = None,
                                   view_confidences: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Black-Litterman Model with investor views
        """
        
        self.logger.info("Starting Black-Litterman Optimization...")
        
        # Calculate implied equilibrium returns
        if market_caps is None:
            # Use equal market cap if not provided
            market_caps = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        
        implied_returns = self._calculate_implied_returns(returns, market_caps)
        
        # Incorporate views
        if views is not None:
            adjusted_returns = self._incorporate_views(implied_returns, returns.cov(), 
                                                     views, view_confidences)
        else:
            adjusted_returns = implied_returns
        
        # Perform mean-variance optimization with adjusted returns
        result = self.mean_variance_optimization(
            pd.DataFrame([adjusted_returns], columns=returns.columns).T,  # Convert to DataFrame
            method='sharpe'
        )
        
        result['optimization_method'] = 'black_litterman'
        return result
    
    def _calculate_implied_returns(self, returns: pd.DataFrame, 
                                 market_caps: pd.Series) -> pd.Series:
        """Calculate implied equilibrium returns"""
        # This is a simplified implementation
        # In practice, you'd use the reverse optimization formula
        cov_matrix = returns.cov()
        market_weights = market_caps / market_caps.sum()
        
        # Implied returns = δ * Σ * w_market
        # Where δ is risk aversion coefficient
        implied_returns = self.risk_aversion * cov_matrix.dot(market_weights)
        return implied_returns
    
    def _incorporate_views(self, implied_returns: pd.Series, cov_matrix: pd.DataFrame,
                         views: Dict[str, float], view_confidences: Dict[str, float]) -> pd.Series:
        """Incorporate investor views into returns"""
        # Simplified implementation
        # In practice, you'd use the Black-Litterman formula
        
        adjusted_returns = implied_returns.copy()
        
        for asset, view_return in views.items():
            if asset in adjusted_returns.index:
                confidence = view_confidences.get(asset, 0.5)
                # Blend implied return with view
                adjusted_returns[asset] = (1 - confidence) * implied_returns[asset] + \
                                         confidence * view_return
        
        return adjusted_returns
    
    def _sharpe_ratio(self, weights: np.ndarray, 
                     expected_returns: pd.Series, 
                     cov_matrix: pd.DataFrame) -> float:
        """Calculate Sharpe ratio for given weights"""
        portfolio_return = self._portfolio_return(weights, expected_returns)
        portfolio_volatility = self._portfolio_volatility(weights, cov_matrix)
        
        if portfolio_volatility == 0:
            return 0
        
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility
    
    def _portfolio_return(self, weights: np.ndarray, expected_returns: pd.Series) -> float:
        """Calculate portfolio return"""
        return np.dot(weights, expected_returns)
    
    def _portfolio_volatility(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def _risk_contribution(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate risk contribution of each asset"""
        portfolio_volatility = self._portfolio_volatility(weights, cov_matrix)
        marginal_contribution = np.dot(cov_matrix, weights)
        risk_contribution = weights * marginal_contribution / portfolio_volatility
        return risk_contribution / portfolio_volatility  # Normalize to percentages
    
    def _calculate_performance_metrics(self, returns: pd.DataFrame):
        """Calculate comprehensive portfolio performance metrics"""
        if self.weights is None:
            return
        
        # Portfolio returns
        portfolio_returns = returns.dot(self.weights)
        
        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Diversification metrics
        diversification_ratio = self._calculate_diversification_ratio(returns.cov())
        concentration_ratio = self._calculate_concentration_ratio()
        
        self.performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'diversification_ratio': diversification_ratio,
            'concentration_ratio': concentration_ratio,
            'number_of_assets': len(self.weights),
            'effective_assets': self._calculate_effective_assets()
        }
    
    def _calculate_diversification_ratio(self, cov_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        if self.weights is None:
            return 0
        
        weighted_vol = np.sqrt(np.diag(cov_matrix)).dot(np.abs(self.weights))
        portfolio_vol = self._portfolio_volatility(self.weights, cov_matrix)
        
        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1
    
    def _calculate_concentration_ratio(self) -> float:
        """Calculate Herfindahl concentration index"""
        if self.weights is None:
            return 0
        return np.sum(self.weights ** 2)
    
    def _calculate_effective_assets(self) -> float:
        """Calculate effective number of assets"""
        concentration = self._calculate_concentration_ratio()
        return 1 / concentration if concentration > 0 else len(self.weights)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        return {
            'optimization_type': 'portfolio_optimization',
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights.to_dict() if self.weights is not None else {},
            'performance_metrics': self.performance_metrics,
            'constraints': {
                'max_allocation': self.max_allocation,
                'min_allocation': self.min_allocation,
                'risk_free_rate': self.risk_free_rate
            },
            'risk_analysis': self._analyze_portfolio_risk()
        }
    
    def _analyze_portfolio_risk(self) -> Dict[str, Any]:
        """Analyze portfolio risk characteristics"""
        if self.weights is None:
            return {}
        
        return {
            'weight_concentration': self._calculate_concentration_ratio(),
            'effective_diversification': self._calculate_effective_assets(),
            'largest_position': self.weights.max() if self.weights is not None else 0,
            'smallest_position': self.weights.min() if self.weights is not None else 0
        }

class MultiStrategyOptimizer:
    """Optimizer for multiple trading strategies"""
    
    def __init__(self):
        self.portfolio_optimizer = PortfolioOptimizer()
        self.logger = logging.getLogger(__name__)
    
    def optimize_strategy_allocation(self,
                                   strategy_returns: Dict[str, pd.Series],
                                   method: str = 'mean_variance') -> Dict[str, Any]:
        """Optimize allocation across multiple trading strategies"""
        
        self.logger.info(f"Optimizing strategy allocation using {method}...")
        
        # Combine strategy returns into DataFrame
        returns_df = pd.DataFrame(strategy_returns)
        
        # Remove any strategies with insufficient data
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            self.logger.error("No valid strategy returns data")
            return {'success': False, 'message': 'No valid data'}
        
        # Perform portfolio optimization
        if method == 'mean_variance':
            result = self.portfolio_optimizer.mean_variance_optimization(returns_df)
        elif method == 'risk_parity':
            result = self.portfolio_optimizer.risk_parity_optimization(returns_df)
        elif method == 'hierarchical_risk_parity':
            result = self.portfolio_optimizer.hierarchical_risk_parity(returns_df)
        else:
            self.logger.error(f"Unknown optimization method: {method}")
            return {'success': False, 'message': f'Unknown method: {method}'}
        
        return result
    
    def calculate_strategy_correlation(self, strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""
        returns_df = pd.DataFrame(strategy_returns).dropna()
        return returns_df.corr()
    
    def analyze_strategy_diversification(self, strategy_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze diversification benefits of strategy combination"""
        
        correlation_matrix = self.calculate_strategy_correlation(strategy_returns)
        returns_df = pd.DataFrame(strategy_returns).dropna()
        
        # Calculate individual strategy performance
        individual_metrics = {}
        for strategy, returns in returns_df.items():
            individual_metrics[strategy] = {
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'volatility': returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(returns)
            }
        
        # Calculate equal-weighted portfolio performance
        equal_weights = pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns)
        equal_weighted_returns = returns_df.dot(equal_weights)
        equal_weighted_metrics = {
            'sharpe_ratio': equal_weighted_returns.mean() / equal_weighted_returns.std() * np.sqrt(252),
            'volatility': equal_weighted_returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(equal_weighted_returns)
        }
        
        return {
            'correlation_matrix': correlation_matrix,
            'individual_metrics': individual_metrics,
            'equal_weighted_metrics': equal_weighted_metrics,
            'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'diversification_potential': self._calculate_diversification_potential(correlation_matrix)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def _calculate_diversification_potential(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate diversification potential score"""
        # Lower average correlation indicates higher diversification potential
        corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
        return 1 - np.mean(np.abs(corr_values))  # 1 = perfect diversification, 0 = no diversification

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Generate sample strategy returns
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    
    # Create correlated strategy returns
    n_strategies = 5
    base_returns = np.random.normal(0.001, 0.02, 1000)
    
    strategy_returns = {}
    for i in range(n_strategies):
        # Create correlated but different returns
        correlation = 0.3 + 0.5 * (i / n_strategies)  # Varying correlation
        strategy_returns[f'Strategy_{i+1}'] = base_returns * correlation + \
                                             np.random.normal(0.001, 0.01, 1000)
    
    returns_df = pd.DataFrame(strategy_returns, index=dates)
    
    # Test portfolio optimization
    multi_optimizer = MultiStrategyOptimizer()
    
    # Analyze strategy diversification
    diversification_analysis = multi_optimizer.analyze_strategy_diversification(strategy_returns)
    print("Diversification Analysis:")
    print(f"Average Correlation: {diversification_analysis['average_correlation']:.4f}")
    print(f"Diversification Potential: {diversification_analysis['diversification_potential']:.4f}")
    
    # Optimize strategy allocation
    optimization_result = multi_optimizer.optimize_strategy_allocation(
        strategy_returns, method='mean_variance'
    )
    
    if optimization_result['success']:
        print("\nPortfolio Optimization Results:")
        print("Optimal Weights:")
        for strategy, weight in optimization_result['weights'].items():
            print(f"  {strategy}: {weight:.4f}")
        
        print("\nPerformance Metrics:")
        for metric, value in optimization_result['performance'].items():
            print(f"  {metric}: {value:.4f}")
    
    # Test different optimization methods
    methods = ['mean_variance', 'risk_parity', 'hierarchical_risk_parity']
    
    print("\nComparison of Optimization Methods:")
    for method in methods:
        result = multi_optimizer.optimize_strategy_allocation(strategy_returns, method=method)
        if result['success']:
            sharpe = result['performance']['sharpe_ratio']
            print(f"  {method}: Sharpe Ratio = {sharpe:.4f}")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """
    Advanced Hyperparameter Tuning for Machine Learning-based Trading Strategies
    Supports Grid Search, Random Search, and Bayesian Optimization
    """
    
    def __init__(self,
                 estimator: Any = None,
                 param_distributions: Dict = None,
                 scoring: str = 'neg_mean_squared_error',
                 cv: int = 5,
                 n_iter: int = 100,
                 random_state: int = 42,
                 n_jobs: int = -1):
        
        self.estimator = estimator
        self.param_distributions = param_distributions or {}
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.cv_results_ = {}
        self.search_history = []
        
        self.logger = self._setup_logging()
        np.random.seed(random_state)
    
    def _setup_logging(self):
        """Setup logging for hyperparameter tuner"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _create_custom_scorer(self, metric: str) -> Callable:
        """Create custom scoring functions for trading strategies"""
        
        def sharpe_scorer(y_true, y_pred):
            """Sharpe Ratio scorer"""
            returns = y_pred  # Assuming predictions are returns
            if len(returns) < 2 or returns.std() == 0:
                return 0
            return returns.mean() / returns.std() * np.sqrt(252)
        
        def calmar_scorer(y_true, y_pred):
            """Calmar Ratio scorer"""
            returns = y_pred
            if len(returns) < 2:
                return 0
            cumulative_returns = (1 + returns).cumprod()
            max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
            annual_return = returns.mean() * 252
            return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        def profit_factor_scorer(y_true, y_pred):
            """Profit Factor scorer"""
            returns = y_pred
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            gross_profit = winning_returns.sum()
            gross_loss = abs(losing_returns.sum())
            return gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        scorers = {
            'sharpe_ratio': make_scorer(sharpe_scorer, greater_is_better=True),
            'calmar_ratio': make_scorer(calmar_scorer, greater_is_better=True),
            'profit_factor': make_scorer(profit_factor_scorer, greater_is_better=True),
            'total_return': make_scorer(lambda y_t, y_p: (1 + y_p).prod() - 1, greater_is_better=True),
            'win_rate': make_scorer(lambda y_t, y_p: (y_p > 0).mean(), greater_is_better=True)
        }
        
        return scorers.get(metric, self.scoring)
    
    def grid_search(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict) -> Dict[str, Any]:
        """Perform grid search over parameter grid"""
        self.logger.info("Starting Grid Search...")
        
        param_combinations = list(ParameterGrid(param_grid))
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        best_score = -np.inf
        best_params = None
        results = []
        
        for i, params in enumerate(param_combinations):
            try:
                # Set parameters
                estimator = self.estimator.set_params(**params)
                
                # Perform time-series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.cv)
                cv_scores = cross_val_score(
                    estimator, X, y, 
                    cv=tscv, 
                    scoring=self._create_custom_scorer(self.scoring),
                    n_jobs=self.n_jobs
                )
                
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                # Store results
                result = {
                    'params': params,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'cv_scores': cv_scores.tolist()
                }
                results.append(result)
                
                # Update best
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(param_combinations)} combinations")
                    
            except Exception as e:
                self.logger.warning(f"Grid search failed for params {params}: {e}")
                continue
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = {
            'grid_search_results': results,
            'best_params': best_params,
            'best_score': best_score
        }
        
        self.logger.info(f"Grid Search completed. Best score: {best_score:.4f}")
        return self.cv_results_
    
    def random_search(self, X: pd.DataFrame, y: pd.Series, n_iter: int = None) -> Dict[str, Any]:
        """Perform random search over parameter distributions"""
        self.logger.info("Starting Random Search...")
        
        n_iter = n_iter or self.n_iter
        param_combinations = list(ParameterSampler(
            self.param_distributions, n_iter, random_state=self.random_state
        ))
        
        self.logger.info(f"Testing {len(param_combinations)} random parameter combinations")
        
        best_score = -np.inf
        best_params = None
        results = []
        
        for i, params in enumerate(param_combinations):
            try:
                # Set parameters
                estimator = self.estimator.set_params(**params)
                
                # Perform time-series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.cv)
                cv_scores = cross_val_score(
                    estimator, X, y,
                    cv=tscv,
                    scoring=self._create_custom_scorer(self.scoring),
                    n_jobs=self.n_jobs
                )
                
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                # Store results
                result = {
                    'params': params,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'cv_scores': cv_scores.tolist()
                }
                results.append(result)
                
                # Update best
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(param_combinations)} iterations")
                    
            except Exception as e:
                self.logger.warning(f"Random search failed for params {params}: {e}")
                continue
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = {
            'random_search_results': results,
            'best_params': best_params,
            'best_score': best_score
        }
        
        self.logger.info(f"Random Search completed. Best score: {best_score:.4f}")
        return self.cv_results_
    
    def bayesian_search(self, X: pd.DataFrame, y: pd.Series, 
                       n_iter: int = None, init_points: int = 10) -> Dict[str, Any]:
        """Perform Bayesian optimization for hyperparameter tuning"""
        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            self.logger.error("scikit-optimize not installed. Install with: pip install scikit-optimize")
            return {}
        
        self.logger.info("Starting Bayesian Search...")
        
        # Convert parameter distributions to skopt space
        search_spaces = {}
        for param, distribution in self.param_distributions.items():
            if isinstance(distribution, list):
                search_spaces[param] = Categorical(distribution)
            elif len(distribution) == 2 and all(isinstance(x, int) for x in distribution):
                search_spaces[param] = Integer(distribution[0], distribution[1])
            elif len(distribution) == 2 and all(isinstance(x, float) for x in distribution):
                search_spaces[param] = Real(distribution[0], distribution[1])
            else:
                self.logger.warning(f"Unsupported distribution for {param}: {distribution}")
                continue
        
        n_iter = n_iter or self.n_iter
        
        # Perform Bayesian optimization
        bayes_search = BayesSearchCV(
            estimator=self.estimator,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=self.cv),
            scoring=self._create_custom_scorer(self.scoring),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_points=1  # Number of parameter sets to evaluate in parallel
        )
        
        bayes_search.fit(X, y)
        
        self.best_params_ = bayes_search.best_params_
        self.best_score_ = bayes_search.best_score_
        
        self.cv_results_ = {
            'bayesian_search_results': {
                'best_params': bayes_search.best_params_,
                'best_score': bayes_search.best_score_,
                'search_history': bayes_search.optimizer_results_
            }
        }
        
        self.logger.info(f"Bayesian Search completed. Best score: {self.best_score_:.4f}")
        return self.cv_results_
    
    def progressive_tuning(self, X: pd.DataFrame, y: pd.Series, 
                          stages: List[Dict] = None) -> Dict[str, Any]:
        """Progressive hyperparameter tuning with multiple stages"""
        
        default_stages = [
            {
                'method': 'random_search',
                'n_iter': 20,
                'param_distributions': self._get_coarse_distributions()
            },
            {
                'method': 'grid_search', 
                'param_grid': self._get_refined_grid(),
                'n_iter': 10
            },
            {
                'method': 'bayesian_search',
                'n_iter': 30,
                'param_distributions': self._get_fine_distributions()
            }
        ]
        
        stages = stages or default_stages
        self.logger.info(f"Starting Progressive Tuning with {len(stages)} stages")
        
        current_best_params = {}
        stage_results = []
        
        for i, stage in enumerate(stages):
            self.logger.info(f"Stage {i + 1}/{len(stages)}: {stage['method']}")
            
            # Update parameter distributions based on previous results
            if current_best_params:
                stage = self._refine_stage_params(stage, current_best_params)
            
            # Execute stage
            method = stage['method']
            if method == 'grid_search':
                result = self.grid_search(X, y, stage.get('param_grid', {}))
            elif method == 'random_search':
                result = self.random_search(X, y, stage.get('n_iter', self.n_iter))
            elif method == 'bayesian_search':
                result = self.bayesian_search(X, y, stage.get('n_iter', self.n_iter))
            else:
                self.logger.warning(f"Unknown method in stage {i + 1}: {method}")
                continue
            
            # Update current best
            if result and 'best_params' in result:
                current_best_params.update(result['best_params'])
                stage_results.append({
                    'stage': i + 1,
                    'method': method,
                    'best_params': result['best_params'],
                    'best_score': result['best_score']
                })
        
        self.best_params_ = current_best_params
        self.cv_results_ = {
            'progressive_tuning_results': stage_results,
            'final_best_params': current_best_params,
            'final_best_score': self.best_score_
        }
        
        return self.cv_results_
    
    def _get_coarse_distributions(self) -> Dict:
        """Get coarse parameter distributions for initial search"""
        coarse_dist = {}
        
        for param, distribution in self.param_distributions.items():
            if isinstance(distribution, list):
                coarse_dist[param] = distribution
            elif len(distribution) == 2:
                # Use wider ranges for coarse search
                if all(isinstance(x, int) for x in distribution):
                    low, high = distribution
                    coarse_dist[param] = list(range(low, high + 1, max(1, (high - low) // 5)))
                elif all(isinstance(x, float) for x in distribution):
                    low, high = distribution
                    coarse_dist[param] = [low, (low + high) / 2, high]
        
        return coarse_dist
    
    def _get_refined_grid(self) -> Dict:
        """Get refined parameter grid based on coarse results"""
        # This would be implemented based on coarse search results
        return self.param_distributions
    
    def _get_fine_distributions(self) -> Dict:
        """Get fine parameter distributions for final tuning"""
        # This would be implemented based on refined results
        return self.param_distributions
    
    def _refine_stage_params(self, stage: Dict, best_params: Dict) -> Dict:
        """Refine stage parameters based on previous best results"""
        refined_stage = stage.copy()
        
        if stage['method'] == 'grid_search':
            # Create tighter grid around best parameters
            grid = {}
            for param, value in best_params.items():
                if param in stage.get('param_grid', {}):
                    if isinstance(value, int):
                        # Create range around best value
                        grid[param] = [max(1, value - 2), value, value + 2]
                    elif isinstance(value, float):
                        grid[param] = [value * 0.8, value, value * 1.2]
            refined_stage['param_grid'] = grid
        
        return refined_stage
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get feature importance from the best estimator"""
        if self.best_params_ is None:
            self.logger.warning("No best parameters found. Run tuning first.")
            return pd.Series()
        
        try:
            # Train best estimator on full data
            best_estimator = self.estimator.set_params(**self.best_params_)
            best_estimator.fit(X, y)
            
            # Get feature importance
            if hasattr(best_estimator, 'feature_importances_'):
                importance = best_estimator.feature_importances_
            elif hasattr(best_estimator, 'coef_'):
                importance = np.abs(best_estimator.coef_)
            else:
                self.logger.warning("Estimator doesn't have feature importance method")
                return pd.Series()
            
            return pd.Series(importance, index=X.columns).sort_values(ascending=False)
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return pd.Series()
    
    def get_tuning_report(self) -> Dict[str, Any]:
        """Generate comprehensive tuning report"""
        return {
            'tuning_method': 'hyperparameter_tuning',
            'timestamp': datetime.now().isoformat(),
            'best_parameters': self.best_params_,
            'best_score': self.best_score_,
            'parameter_distributions': self.param_distributions,
            'cross_validation_settings': {
                'cv_splits': self.cv,
                'scoring': self.scoring,
                'n_jobs': self.n_jobs
            },
            'performance_analysis': self._analyze_performance()
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze tuning performance and convergence"""
        if not self.cv_results_:
            return {}
        
        # Extract scores from different search methods
        scores = []
        if 'grid_search_results' in self.cv_results_:
            scores.extend([r['mean_score'] for r in self.cv_results_['grid_search_results']])
        if 'random_search_results' in self.cv_results_:
            scores.extend([r['mean_score'] for r in self.cv_results_['random_search_results']])
        
        if not scores:
            return {}
        
        return {
            'score_statistics': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            },
            'improvement_over_default': self._calculate_improvement(),
            'parameter_sensitivity': self._analyze_parameter_sensitivity()
        }
    
    def _calculate_improvement(self) -> float:
        """Calculate improvement over default parameters"""
        # This would compare with default estimator performance
        return 0.0  # Placeholder
    
    def _analyze_parameter_sensitivity(self) -> Dict[str, Any]:
        """Analyze sensitivity of parameters to performance"""
        # This would analyze how each parameter affects the score
        return {}  # Placeholder

class MLStrategyTuner:
    """Specialized tuner for machine learning trading strategies"""
    
    def __init__(self):
        self.tuner = HyperparameterTuner()
        self.logger = logging.getLogger(__name__)
    
    def tune_classification_strategy(self, X: pd.DataFrame, y: pd.Series,
                                   model, param_distributions: Dict) -> Dict[str, Any]:
        """Tune classification-based trading strategy"""
        
        self.tuner.estimator = model
        self.tuner.param_distributions = param_distributions
        self.tuner.scoring = 'accuracy'  # Or custom trading metric
        
        return self.tuner.random_search(X, y)
    
    def tune_regression_strategy(self, X: pd.DataFrame, y: pd.Series,
                               model, param_distributions: Dict) -> Dict[str, Any]:
        """Tune regression-based trading strategy"""
        
        self.tuner.estimator = model
        self.tuner.param_distributions = param_distributions
        self.tuner.scoring = 'neg_mean_squared_error'
        
        return self.tuner.random_search(X, y)
    
    def create_trading_features(self, data: pd.DataFrame, 
                              lookback_periods: List[int] = None) -> pd.DataFrame:
        """Create features for ML-based trading strategies"""
        
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 50]
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['volatility'] = data['close'].pct_change().rolling(20).std()
        
        for period in lookback_periods:
            # Moving averages
            features[f'ma_{period}'] = data['close'].rolling(period).mean()
            features[f'ma_ratio_{period}'] = data['close'] / features[f'ma_{period}']
            
            # High-Low ranges
            features[f'high_low_ratio_{period}'] = (
                data['high'].rolling(period).max() / 
                data['low'].rolling(period).min()
            )
            
            # Volume features
            features[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = (
                data['volume'] / features[f'volume_ma_{period}']
            )
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd'] = self._calculate_macd(data['close'])
        features['bollinger_position'] = self._calculate_bollinger_position(data)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd
    
    def _calculate_bollinger_position(self, data: pd.DataFrame, 
                                    period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        ma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        return (data['close'] - lower_band) / (upper_band - lower_band)

if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
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
    
    # Create features and targets
    ml_tuner = MLStrategyTuner()
    features = ml_tuner.create_trading_features(sample_data)
    
    # Create binary target (1 if next return is positive, 0 otherwise)
    targets = (features['returns'].shift(-1) > 0).astype(int)
    features = features[:-1]  # Remove last row due to target shift
    targets = targets[:-1].dropna()
    
    # Align indices
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]
    
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Define model and parameter distributions
    model = RandomForestClassifier(random_state=42)
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Perform hyperparameter tuning
    tuner = HyperparameterTuner(
        estimator=model,
        param_distributions=param_distributions,
        scoring='accuracy',
        cv=5,
        n_iter=20
    )
    
    results = tuner.random_search(features, targets)
    
    print("Hyperparameter Tuning Results:")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best Score: {results['best_score']:.4f}")
    
    # Get feature importance
    importance = tuner.get_feature_importance(features, targets)
    print("\nTop 10 Feature Importances:")
    print(importance.head(10))
    
    # Generate tuning report
    report = tuner.get_tuning_report()
    print(f"\nTuning Report: {report['performance_analysis']}")

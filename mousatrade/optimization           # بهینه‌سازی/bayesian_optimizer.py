import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BayesianOptimizer:
    """
    Bayesian Optimization for trading strategy parameters
    Uses Gaussian Processes and Expected Improvement
    """
    
    def __init__(self,
                 n_iter: int = 100,
                 init_points: int = 10,
                 acq_func: str = 'ei',
                 random_state: int = 42,
                 objective_function: Callable = None):
        
        self.n_iter = n_iter
        self.init_points = init_points
        self.acq_func = acq_func
        self.random_state = random_state
        self.objective_function = objective_function
        
        self.X_obs = []  # Observed parameters
        self.y_obs = []  # Observed objective values
        self.gp = None
        self.parameter_space = None
        self.best_parameters = None
        self.best_value = -np.inf
        
        self.logger = self._setup_logging()
        np.random.seed(random_state)
    
    def _setup_logging(self):
        """Setup logging for Bayesian optimizer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _initialize_parameters(self, parameter_space: Dict[str, Tuple]) -> List[Dict]:
        """Initialize parameters with random samples"""
        samples = []
        
        for _ in range(self.init_points):
            sample = {}
            for param, (min_val, max_val, param_type) in parameter_space.items():
                if param_type == 'int':
                    sample[param] = np.random.randint(min_val, max_val + 1)
                elif param_type == 'float':
                    sample[param] = np.random.uniform(min_val, max_val)
                elif param_type == 'categorical':
                    sample[param] = np.random.choice(min_val)
            samples.append(sample)
        
        return samples
    
    def _parameter_to_vector(self, params: Dict, parameter_space: Dict) -> np.ndarray:
        """Convert parameters to feature vector"""
        vector = []
        
        for param, (min_val, max_val, param_type) in parameter_space.items():
            value = params[param]
            
            if param_type in ['int', 'float']:
                # Normalize to [0, 1]
                normalized = (value - min_val) / (max_val - min_val)
                vector.append(normalized)
            elif param_type == 'categorical':
                # One-hot encoding for categorical parameters
                categories = min_val
                one_hot = [1 if value == cat else 0 for cat in categories]
                vector.extend(one_hot)
        
        return np.array(vector)
    
    def _vector_to_parameter(self, vector: np.ndarray, parameter_space: Dict) -> Dict:
        """Convert feature vector back to parameters"""
        params = {}
        idx = 0
        
        for param, (min_val, max_val, param_type) in parameter_space.items():
            if param_type in ['int', 'float']:
                # Denormalize from [0, 1]
                denormalized = vector[idx] * (max_val - min_val) + min_val
                if param_type == 'int':
                    params[param] = int(round(denormalized))
                else:
                    params[param] = denormalized
                idx += 1
            elif param_type == 'categorical':
                categories = min_val
                one_hot = vector[idx:idx + len(categories)]
                category_idx = np.argmax(one_hot)
                params[param] = categories[category_idx]
                idx += len(categories)
        
        return params
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Expected Improvement acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if np.max(sigma) == 0:
            return np.zeros_like(mu)
        
        mu_sample_opt = np.max(self.y_obs)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _probability_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Probability of Improvement acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if np.max(sigma) == 0:
            return np.zeros_like(mu)
        
        mu_sample_opt = np.max(self.y_obs)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            pi = norm.cdf(Z)
        
        return pi
    
    def _upper_confidence_bound(self, X: np.ndarray, kappa: float = 2.576) -> np.ndarray:
        """Calculate Upper Confidence Bound acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + kappa * sigma
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Select and compute acquisition function"""
        if self.acq_func == 'ei':
            return self._expected_improvement(X)
        elif self.acq_func == 'pi':
            return self._probability_improvement(X)
        elif self.acq_func == 'ucb':
            return self._upper_confidence_bound(X)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acq_func}")
    
    def _propose_next_point(self, n_candidates: int = 1000) -> Dict:
        """Propose next point to evaluate using acquisition function"""
        # Generate candidate points
        candidates = []
        for _ in range(n_candidates):
            candidate = {}
            for param, (min_val, max_val, param_type) in self.parameter_space.items():
                if param_type == 'int':
                    candidate[param] = np.random.randint(min_val, max_val + 1)
                elif param_type == 'float':
                    candidate[param] = np.random.uniform(min_val, max_val)
                elif param_type == 'categorical':
                    candidate[param] = np.random.choice(min_val)
            candidates.append(candidate)
        
        # Convert to feature vectors
        X_candidates = np.array([self._parameter_to_vector(cand, self.parameter_space) 
                               for cand in candidates])
        
        # Calculate acquisition function values
        acq_values = self._acquisition_function(X_candidates)
        
        # Select candidate with highest acquisition value
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]
    
    def _fit_gaussian_process(self):
        """Fit Gaussian Process model to observed data"""
        if len(self.X_obs) < 2:
            return
        
        X = np.array([self._parameter_to_vector(x, self.parameter_space) for x in self.X_obs])
        y = np.array(self.y_obs)
        
        # Define kernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        
        # Fit GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )
        
        self.gp.fit(X, y)
    
    def optimize(self,
                 parameter_space: Dict[str, Tuple],
                 data: pd.DataFrame,
                 **kwargs) -> Dict[str, Any]:
        """Main Bayesian optimization method"""
        
        self.logger.info("Starting Bayesian optimization...")
        self.parameter_space = parameter_space
        
        # Initial random sampling
        self.logger.info(f"Performing {self.init_points} initial random evaluations...")
        initial_samples = self._initialize_parameters(parameter_space)
        
        for sample in initial_samples:
            fitness = self.objective_function(sample, data, **kwargs)
            self.X_obs.append(sample)
            self.y_obs.append(fitness)
            
            # Update best
            if fitness > self.best_value:
                self.best_value = fitness
                self.best_parameters = sample.copy()
        
        # Bayesian optimization loop
        self.logger.info(f"Starting {self.n_iter} Bayesian optimization iterations...")
        
        for iteration in range(self.n_iter):
            # Fit Gaussian Process
            self._fit_gaussian_process()
            
            # Propose next point
            next_point = self._propose_next_point()
            
            # Evaluate objective function
            fitness = self.objective_function(next_point, data, **kwargs)
            
            # Update observations
            self.X_obs.append(next_point)
            self.y_obs.append(fitness)
            
            # Update best
            if fitness > self.best_value:
                self.best_value = fitness
                self.best_parameters = next_point.copy()
            
            # Log progress
            if (iteration + 1) % 10 == 0:
                self.logger.info(
                    f"Iteration {iteration + 1}/{self.n_iter}: "
                    f"Best Fitness = {self.best_value:.4f}, "
                    f"Current Fitness = {fitness:.4f}"
                )
        
        self.logger.info("Bayesian optimization completed!")
        
        return {
            'best_parameters': self.best_parameters,
            'best_fitness': self.best_value,
            'evaluation_history': list(zip(self.X_obs, self.y_obs)),
            'total_evaluations': len(self.X_obs)
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        return {
            'optimization_type': 'bayesian_optimization',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'n_iter': self.n_iter,
                'init_points': self.init_points,
                'acq_func': self.acq_func,
                'random_state': self.random_state
            },
            'results': {
                'best_parameters': self.best_parameters,
                'best_fitness': self.best_value,
                'total_evaluations': len(self.X_obs)
            },
            'convergence_analysis': self._analyze_convergence()
        }
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence of Bayesian optimization"""
        if len(self.y_obs) < 2:
            return {}
        
        # Calculate running maximum
        running_max = [self.y_obs[0]]
        for i in range(1, len(self.y_obs)):
            running_max.append(max(running_max[-1], self.y_obs[i]))
        
        improvements = [running_max[i] - running_max[i-1] for i in range(1, len(running_max))]
        
        return {
            'final_fitness': running_max[-1],
            'total_improvement': running_max[-1] - running_max[0],
            'avg_improvement': np.mean(improvements) if improvements else 0,
            'convergence_iteration': self._find_convergence_point(running_max)
        }
    
    def _find_convergence_point(self, running_max: List[float], window: int = 5) -> int:
        """Find iteration where convergence occurred"""
        if len(running_max) < window * 2:
            return len(running_max) - 1
        
        for i in range(len(running_max) - window):
            current_max = running_max[i]
            next_max = running_max[i + window]
            
            # Check if improvement is below threshold
            improvement = next_max - current_max
            if improvement < 0.001:  # Convergence threshold
                return i
        
        return len(running_max) - 1
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Estimate parameter importance using GP kernel analysis"""
        if self.gp is None or len(self.X_obs) < 2:
            return {}
        
        try:
            # Get kernel length scales
            kernel = self.gp.kernel_
            if hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
                length_scales = kernel.k2.length_scale
                
                # Convert to importance scores (inverse of length scale)
                if isinstance(length_scales, np.ndarray):
                    importance = 1.0 / length_scales
                    
                    # Map back to parameters
                    param_importance = {}
                    idx = 0
                    
                    for param, (min_val, max_val, param_type) in self.parameter_space.items():
                        if param_type in ['int', 'float']:
                            param_importance[param] = float(importance[idx])
                            idx += 1
                        elif param_type == 'categorical':
                            # For categorical, take average of one-hot dimensions
                            cat_importance = importance[idx:idx + len(min_val)]
                            param_importance[param] = float(np.mean(cat_importance))
                            idx += len(min_val)
                    
                    # Normalize importance scores
                    total_importance = sum(param_importance.values())
                    if total_importance > 0:
                        param_importance = {k: v/total_importance for k, v in param_importance.items()}
                    
                    return param_importance
            
        except Exception as e:
            self.logger.warning(f"Parameter importance analysis failed: {e}")
        
        return {}

class AdvancedBayesianOptimizer(BayesianOptimizer):
    """Enhanced Bayesian Optimizer with advanced features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_importance_history = []
    
    def optimize_with_priors(self,
                           parameter_space: Dict[str, Tuple],
                           priors: Dict[str, Any],
                           data: pd.DataFrame,
                           **kwargs) -> Dict[str, Any]:
        """Optimize with prior knowledge about parameters"""
        
        self.logger.info("Starting Bayesian optimization with priors...")
        self.parameter_space = parameter_space
        
        # Incorporate priors in initial sampling
        initial_samples = self._initialize_with_priors(parameter_space, priors)
        
        # Evaluate initial samples
        for sample in initial_samples:
            fitness = self.objective_function(sample, data, **kwargs)
            self.X_obs.append(sample)
            self.y_obs.append(fitness)
            
            if fitness > self.best_value:
                self.best_value = fitness
                self.best_parameters = sample.copy()
        
        # Continue with standard Bayesian optimization
        return self.optimize(parameter_space, data, **kwargs)
    
    def _initialize_with_priors(self, 
                              parameter_space: Dict[str, Tuple],
                              priors: Dict[str, Any]) -> List[Dict]:
        """Initialize samples incorporating prior knowledge"""
        samples = []
        
        # Sample around prior means
        for _ in range(self.init_points):
            sample = {}
            for param, (min_val, max_val, param_type) in parameter_space.items():
                if param in priors:
                    prior = priors[param]
                    
                    if param_type == 'int':
                        # Sample from normal distribution around prior
                        mean = prior.get('mean', (min_val + max_val) / 2)
                        std = prior.get('std', (max_val - min_val) / 6)
                        value = int(np.clip(np.random.normal(mean, std), min_val, max_val))
                    elif param_type == 'float':
                        mean = prior.get('mean', (min_val + max_val) / 2)
                        std = prior.get('std', (max_val - min_val) / 6)
                        value = np.clip(np.random.normal(mean, std), min_val, max_val)
                    elif param_type == 'categorical':
                        # Use prior probabilities
                        probabilities = prior.get('probabilities', 
                                                [1/len(min_val)] * len(min_val))
                        value = np.random.choice(min_val, p=probabilities)
                    
                    sample[param] = value
                else:
                    # Random sampling for parameters without priors
                    if param_type == 'int':
                        sample[param] = np.random.randint(min_val, max_val + 1)
                    elif param_type == 'float':
                        sample[param] = np.random.uniform(min_val, max_val)
                    elif param_type == 'categorical':
                        sample[param] = np.random.choice(min_val)
            
            samples.append(sample)
        
        return samples

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 10, 500),
        'high': np.random.normal(105, 10, 500),
        'low': np.random.normal(95, 10, 500),
        'close': 100 + np.cumsum(np.random.normal(0, 1, 500)),
        'volume': np.random.normal(1000, 200, 500)
    }, index=dates)
    
    # Define objective function
    def sample_objective(params, data):
        """Sample objective function - maximize returns with low volatility"""
        try:
            # Simple moving average strategy
            fast_ma = data['close'].rolling(window=params['fast_window']).mean()
            slow_ma = data['close'].rolling(window=params['slow_window']).mean()
            
            signals = np.where(fast_ma > slow_ma, 1, -1)
            positions = pd.Series(signals).shift(1)
            
            returns = data['close'].pct_change()
            strategy_returns = positions * returns
            
            # Remove NaN
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) < 10:
                return -np.inf
            
            # Compound return with volatility penalty
            total_return = (1 + strategy_returns).prod() - 1
            volatility = strategy_returns.std()
            
            if volatility == 0:
                return -np.inf
            
            # Sharpe-like ratio
            score = total_return / volatility
            
            return score if not np.isnan(score) else -np.inf
            
        except Exception as e:
            return -np.inf
    
    # Define parameter space
    parameter_space = {
        'fast_window': (5, 50, 'int'),
        'slow_window': (20, 200, 'int')
    }
    
    # Run Bayesian optimization
    bayesian_opt = BayesianOptimizer(
        n_iter=20,
        init_points=5,
        acq_func='ei',
        objective_function=sample_objective
    )
    
    results = bayesian_opt.optimize(parameter_space, sample_data)
    
    print("Bayesian Optimization Results:")
    print(f"Best Parameters: {results['best_parameters']}")
    print(f"Best Fitness: {results['best_fitness']:.4f}")
    
    # Get parameter importance
    importance = bayesian_opt.get_parameter_importance()
    print(f"Parameter Importance: {importance}")
    
    # Generate report
    report = bayesian_opt.get_optimization_report()
    print(f"Convergence Analysis: {report['convergence_analysis']}")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
import random
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

class GeneticOptimizer:
    """
    Genetic Algorithm for optimizing trading strategy parameters
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_count: int = 2,
                 objective_function: Callable = None):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.objective_function = objective_function
        
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for genetic optimizer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def initialize_population(self, parameter_space: Dict[str, Tuple]) -> List[Dict]:
        """Initialize population with random parameters"""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val, param_type) in parameter_space.items():
                if param_type == 'int':
                    individual[param] = random.randint(int(min_val), int(max_val))
                elif param_type == 'float':
                    individual[param] = random.uniform(min_val, max_val)
                elif param_type == 'categorical':
                    individual[param] = random.choice(min_val)  # min_val is list of categories
            population.append(individual)
        
        return population
    
    def evaluate_fitness(self, individual: Dict, *args, **kwargs) -> float:
        """Evaluate fitness of an individual"""
        try:
            if self.objective_function:
                return self.objective_function(individual, *args, **kwargs)
            else:
                # Default objective: sharpe ratio
                return self._default_objective(individual, *args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed: {e}")
            return -np.inf
    
    def _default_objective(self, individual: Dict, returns: pd.Series) -> float:
        """Default objective function - Sharpe Ratio"""
        if len(returns) == 0:
            return -np.inf
        
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        return sharpe_ratio
    
    def selection(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Tournament selection"""
        selected = []
        
        # Elitism - keep best individuals
        elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
        for idx in elite_indices:
            selected.append(population[idx])
        
        # Tournament selection for the rest
        while len(selected) < self.population_size:
            # Randomly select 3 individuals
            tournament_indices = random.sample(range(len(population)), 3)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select the best one
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        # Select crossover point
        params = list(parent1.keys())
        crossover_point = random.randint(1, len(params) - 1)
        
        # Swap parameters after crossover point
        for i in range(crossover_point, len(params)):
            param = params[i]
            child1[param], child2[param] = child2[param], child1[param]
        
        return child1, child2
    
    def mutate(self, individual: Dict, parameter_space: Dict) -> Dict:
        """Mutate an individual"""
        mutated = individual.copy()
        
        for param, (min_val, max_val, param_type) in parameter_space.items():
            if random.random() < self.mutation_rate:
                if param_type == 'int':
                    mutated[param] = random.randint(int(min_val), int(max_val))
                elif param_type == 'float':
                    mutated[param] = random.uniform(min_val, max_val)
                elif param_type == 'categorical':
                    mutated[param] = random.choice(min_val)
        
        return mutated
    
    def optimize(self, 
                 parameter_space: Dict[str, Tuple],
                 data: pd.DataFrame,
                 parallel: bool = True,
                 **kwargs) -> Dict[str, Any]:
        """Main optimization method"""
        
        self.logger.info("Starting genetic optimization...")
        self.population = self.initialize_population(parameter_space)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self._evaluate_population_fitness(
                self.population, data, parallel, **kwargs
            )
            
            # Update best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores)
            })
            
            # Selection
            selected = self.selection(self.population, fitness_scores)
            
            # Crossover and Mutation
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                new_population.append(self.mutate(child1, parameter_space))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2, parameter_space))
            
            self.population = new_population
            
            # Log progress
            if generation % 10 == 0:
                self.logger.info(
                    f"Generation {generation}: Best Fitness = {self.best_fitness:.4f}, "
                    f"Avg Fitness = {np.mean(fitness_scores):.4f}"
                )
        
        self.logger.info("Genetic optimization completed!")
        
        return {
            'best_parameters': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'final_population': self.population
        }
    
    def _evaluate_population_fitness(self, 
                                   population: List[Dict], 
                                   data: pd.DataFrame,
                                   parallel: bool = True,
                                   **kwargs) -> List[float]:
        """Evaluate fitness for entire population"""
        if parallel:
            return self._evaluate_parallel(population, data, **kwargs)
        else:
            return [self.evaluate_fitness(ind, data, **kwargs) for ind in population]
    
    def _evaluate_parallel(self, 
                          population: List[Dict],
                          data: pd.DataFrame,
                          **kwargs) -> List[float]:
        """Evaluate fitness in parallel"""
        fitness_scores = []
        
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.evaluate_fitness, ind, data, **kwargs): i
                for i, ind in enumerate(population)
            }
            
            for future in as_completed(futures):
                fitness_scores.append(future.result())
        
        return fitness_scores
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        return {
            'optimization_type': 'genetic_algorithm',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_count': self.elitism_count
            },
            'results': {
                'best_parameters': self.best_individual,
                'best_fitness': self.best_fitness,
                'total_evaluations': self.population_size * self.generations
            },
            'convergence_analysis': self._analyze_convergence()
        }
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence of the genetic algorithm"""
        if not self.fitness_history:
            return {}
        
        fitness_values = [fh['best_fitness'] for fh in self.fitness_history]
        
        return {
            'final_fitness': fitness_values[-1],
            'improvement': fitness_values[-1] - fitness_values[0],
            'convergence_generation': self._find_convergence_point(fitness_values),
            'fitness_std': np.std(fitness_values)
        }
    
    def _find_convergence_point(self, fitness_values: List[float], window: int = 10) -> int:
        """Find generation where convergence occurred"""
        if len(fitness_values) < window * 2:
            return len(fitness_values) - 1
        
        for i in range(len(fitness_values) - window):
            current_window = fitness_values[i:i+window]
            next_window = fitness_values[i+window:i+2*window]
            
            if len(next_window) < window:
                break
            
            # Check if improvement is below threshold
            improvement = np.mean(next_window) - np.mean(current_window)
            if improvement < 0.001:  # Convergence threshold
                return i + window
        
        return len(fitness_values) - 1

# Example usage and strategy optimization
class StrategyOptimizer:
    """Wrapper for optimizing specific trading strategies"""
    
    def __init__(self):
        self.genetic_optimizer = GeneticOptimizer()
        self.logger = logging.getLogger(__name__)
    
    def optimize_moving_average_strategy(self, 
                                       data: pd.DataFrame,
                                       fast_window_range: Tuple = (5, 50),
                                       slow_window_range: Tuple = (20, 200),
                                       **kwargs) -> Dict[str, Any]:
        """Optimize moving average crossover strategy"""
        
        parameter_space = {
            'fast_window': (*fast_window_range, 'int'),
            'slow_window': (*slow_window_range, 'int'),
            'volume_filter': ([True, False], [True, False], 'categorical')
        }
        
        return self.genetic_optimizer.optimize(
            parameter_space=parameter_space,
            data=data,
            objective_function=self._ma_strategy_objective,
            **kwargs
        )
    
    def _ma_strategy_objective(self, params: Dict, data: pd.DataFrame) -> float:
        """Objective function for MA strategy"""
        try:
            fast_window = params['fast_window']
            slow_window = params['slow_window']
            
            if fast_window >= slow_window:
                return -np.inf
            
            # Calculate moving averages
            data = data.copy()
            data['fast_ma'] = data['close'].rolling(window=fast_window).mean()
            data['slow_ma'] = data['close'].rolling(window=slow_window).mean()
            
            # Generate signals
            data['signal'] = 0
            data['signal'] = np.where(data['fast_ma'] > data['slow_ma'], 1, -1)
            data['position'] = data['signal'].shift(1)
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['position'] * data['returns']
            
            # Remove NaN values
            strategy_returns = data['strategy_returns'].dropna()
            
            if len(strategy_returns) == 0:
                return -np.inf
            
            # Calculate Sharpe ratio
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            # Penalize for too few trades
            trades = (data['position'].diff() != 0).sum()
            if trades < len(data) * 0.01:  # At least 1% of periods should have trades
                sharpe_ratio *= 0.5
            
            return sharpe_ratio if not np.isnan(sharpe_ratio) else -np.inf
            
        except Exception as e:
            self.logger.warning(f"MA strategy evaluation failed: {e}")
            return -np.inf
    
    def optimize_rsi_strategy(self,
                            data: pd.DataFrame,
                            rsi_period_range: Tuple = (5, 30),
                            oversold_range: Tuple = (20, 40),
                            overbought_range: Tuple = (60, 80),
                            **kwargs) -> Dict[str, Any]:
        """Optimize RSI strategy parameters"""
        
        parameter_space = {
            'rsi_period': (*rsi_period_range, 'int'),
            'oversold_level': (*oversold_range, 'int'),
            'overbought_level': (*overbought_range, 'int')
        }
        
        return self.genetic_optimizer.optimize(
            parameter_space=parameter_space,
            data=data,
            objective_function=self._rsi_strategy_objective,
            **kwargs
        )
    
    def _rsi_strategy_objective(self, params: Dict, data: pd.DataFrame) -> float:
        """Objective function for RSI strategy"""
        try:
            from scripts.analysis.technical_analysis import TechnicalAnalyzer
            
            analyzer = TechnicalAnalyzer()
            data = analyzer.calculate_all_indicators(data)
            
            rsi_period = params['rsi_period']
            oversold = params['oversold_level']
            overbought = params['overbought_level']
            
            if oversold >= overbought:
                return -np.inf
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Generate signals
            data['signal'] = 0
            data['signal'] = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
            data['position'] = data['signal'].shift(1)
            
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['position'] * data['returns']
            
            strategy_returns = data['strategy_returns'].dropna()
            
            if len(strategy_returns) == 0:
                return -np.inf
            
            # Calculate Sortino ratio (focus on downside risk)
            negative_returns = strategy_returns[strategy_returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
            
            if downside_std == 0:
                sortino_ratio = strategy_returns.mean() * np.sqrt(252)
            else:
                sortino_ratio = strategy_returns.mean() / downside_std * np.sqrt(252)
            
            return sortino_ratio if not np.isnan(sortino_ratio) else -np.inf
            
        except Exception as e:
            self.logger.warning(f"RSI strategy evaluation failed: {e}")
            return -np.inf

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 10, 1000),
        'high': np.random.normal(105, 10, 1000),
        'low': np.random.normal(95, 10, 1000),
        'close': np.random.normal(100, 10, 1000),
        'volume': np.random.normal(1000, 200, 1000)
    }, index=dates)
    
    # Test genetic optimizer
    optimizer = StrategyOptimizer()
    
    # Optimize MA strategy
    ma_results = optimizer.optimize_moving_average_strategy(
        sample_data,
        population_size=20,
        generations=10  # Small for testing
    )
    
    print("MA Strategy Optimization Results:")
    print(f"Best Parameters: {ma_results['best_parameters']}")
    print(f"Best Fitness: {ma_results['best_fitness']:.4f}")
    
    # Optimize RSI strategy
    rsi_results = optimizer.optimize_rsi_strategy(
        sample_data,
        population_size=20,
        generations=10
    )
    
    print("\nRSI Strategy Optimization Results:")
    print(f"Best Parameters: {rsi_results['best_parameters']}")
    print(f"Best Fitness: {rsi_results['best_fitness']:.4f}")

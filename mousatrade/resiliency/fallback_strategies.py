from typing import Callable, Any, List, Optional, Dict
from abc import ABC, abstractmethod

class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies"""
    
    @abstractmethod
    def execute(self, main_func: Callable, fallback_funcs: List[Callable], *args, **kwargs) -> Any:
        pass

class SequentialFallback(FallbackStrategy):
    """
    Execute fallback functions sequentially until one succeeds
    """
    
    def execute(self, main_func: Callable, fallback_funcs: List[Callable], *args, **kwargs) -> Any:
        all_functions = [main_func] + fallback_funcs
        
        for i, func in enumerate(all_functions):
            try:
                result = func(*args, **kwargs)
                if i > 0:  # If fallback was used
                    print(f"🔄 Fallback strategy {i} succeeded for {func.__name__}")
                return result
            except Exception as e:
                print(f"❌ Strategy {i} failed: {str(e)}")
                if i == len(all_functions) - 1:  # Last attempt
                    raise FallbackExhaustedError(
                        f"All fallback strategies exhausted for {main_func.__name__}"
                    ) from e

class BestEffortFallback(FallbackStrategy):
    """
    Try all strategies and return the first successful result
    """
    
    def execute(self, main_func: Callable, fallback_funcs: List[Callable], *args, **kwargs) -> Any:
        all_functions = [main_func] + fallback_funcs
        last_exception = None
        
        for func in all_functions:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                print(f"❌ Strategy failed: {func.__name__} - {str(e)}")
        
        raise FallbackExhaustedError(
            f"No fallback strategy succeeded for {main_func.__name__}"
        ) from last_exception

class FallbackStrategies:
    """
    Main fallback strategies coordinator for trading operations
    """
    
    def __init__(self):
        self.sequential = SequentialFallback()
        self.best_effort = BestEffortFallback()
    
    def trading_execution(self, symbol: str, order_type: str, amount: float):
        """Fallback strategies for trade execution"""
        def primary_exchange():
            # Your main exchange execution logic
            pass
            
        def secondary_exchange():
            # Fallback to secondary exchange
            pass
            
        def market_maker_strategy():
            # Alternative execution strategy
            pass
        
        strategies = [secondary_exchange, market_maker_strategy]
        return self.sequential.execute(primary_exchange, strategies, symbol, order_type, amount)
    
    def price_feed(self, symbol: str):
        """Fallback strategies for price data"""
        def primary_feed():
            # Main price feed
            pass
            
        def secondary_feed():
            # Backup price feed
            pass
            
        def cached_data():
            # Use cached data as last resort
            pass
        
        strategies = [secondary_feed, cached_data]
        return self.sequential.execute(primary_feed, strategies, symbol)

class FallbackExhaustedError(Exception):
    """Exception when all fallback strategies fail"""
    pass

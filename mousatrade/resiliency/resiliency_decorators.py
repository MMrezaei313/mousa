from typing import Callable, Any
from functools import wraps
from .circuit_breaker import CircuitBreakerRegistry
from .retry_mechanism import TradingRetryConfig
from .fallback_strategies import FallbackStrategies
from .error_handler import catch_trading_errors

def resilient_trade_execution(exchange_name: str = "default"):
    """
    Comprehensive resiliency decorator for trading operations
    Combines circuit breaker, retry, fallback, and error handling
    """
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        @catch_trading_errors({"operation": func.__name__})
        def wrapper(*args, **kwargs):
            # Get circuit breaker for this exchange
            circuit_breaker = CircuitBreakerRegistry.get_exchange_breaker(exchange_name)
            
            # Get retry mechanism
            retry_mechanism = TradingRetryConfig.exchange_operation()
            
            # Define fallback strategies
            fallback = FallbackStrategies()
            
            # Execute with all resiliency patterns
            def execute_with_resiliency():
                return circuit_breaker.call(func, *args, **kwargs)
            
            return retry_mechanism.execute(execute_with_resiliency)
            
        return wrapper
    return decorator

def resilient_data_feed(service_name: str = "data_feed"):
    """Resiliency decorator for data feed operations"""
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        @catch_trading_errors({"operation": func.__name__})
        def wrapper(*args, **kwargs):
            circuit_breaker = CircuitBreakerRegistry.get_api_breaker(service_name)
            retry_mechanism = TradingRetryConfig.data_feed_operation()
            
            def execute_with_resiliency():
                return circuit_breaker.call(func, *args, **kwargs)
            
            return retry_mechanism.execute(execute_with_resiliency)
            
        return wrapper
    return decorator

"""
Resiliency Module for Mousa Trading System
Advanced error handling and fault tolerance patterns
"""

from .circuit_breaker import TradingCircuitBreaker, CircuitBreakerRegistry, CircuitBreakerError
from .retry_mechanism import RetryMechanism, RetryExhaustedError, TradingRetryConfig
from .fallback_strategies import FallbackStrategies, FallbackExhaustedError, SequentialFallback, BestEffortFallback
from .health_check import HealthChecker, HealthStatus, HealthCheckResult
from .error_handler import ErrorHandler, catch_trading_errors
from .resiliency_decorators import resilient_trade_execution, resilient_data_feed

# Config imports
from .config import (
    CircuitBreakerConfig,
    RetryConfig, 
    HealthCheckConfig,
    FallbackConfig,
    ResiliencyConfig,
    DEFAULT_CONFIG,
    PRODUCTION_CONFIG,
    DEVELOPMENT_CONFIG,
    get_config
)

__version__ = "1.0.0"
__author__ = "Mousa Trading Team"

__all__ = [
    # Circuit Breaker
    'TradingCircuitBreaker',
    'CircuitBreakerRegistry', 
    'CircuitBreakerError',
    
    # Retry Mechanism
    'RetryMechanism',
    'RetryExhaustedError',
    'TradingRetryConfig',
    
    # Fallback Strategies
    'FallbackStrategies',
    'FallbackExhaustedError', 
    'SequentialFallback',
    'BestEffortFallback',
    
    # Health Check
    'HealthChecker',
    'HealthStatus',
    'HealthCheckResult',
    
    # Error Handling
    'ErrorHandler',
    'catch_trading_errors',
    
    # Decorators
    'resilient_trade_execution',
    'resilient_data_feed',
    
    # Configuration
    'CircuitBreakerConfig',
    'RetryConfig',
    'HealthCheckConfig', 
    'FallbackConfig',
    'ResiliencyConfig',
    'DEFAULT_CONFIG',
    'PRODUCTION_CONFIG',
    'DEVELOPMENT_CONFIG',
    'get_config'
]

# Initialize default error handler on module import
_default_error_handler = ErrorHandler()

def handle_error(error: Exception, context: dict = None):
    """Global error handling function"""
    _default_error_handler.handle_trading_error(error, context)

def get_global_config() -> ResiliencyConfig:
    """Get global resiliency configuration"""
    return DEFAULT_CONFIG

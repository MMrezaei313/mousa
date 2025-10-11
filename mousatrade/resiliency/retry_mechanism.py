import time
import random
from typing import Callable, Any, Optional, Dict
from functools import wraps

class RetryMechanism:
    """
    Retry mechanism with exponential backoff and jitter
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, jitter: bool = True,
                 retry_on_exceptions: tuple = (Exception,)):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on_exceptions = retry_on_exceptions
        
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self._wait_before_retry(attempt)
                
                return func(*args, **kwargs)
                
            except self.retry_on_exceptions as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                    
                print(f"🔄 Retry {attempt + 1}/{self.max_retries} for {func.__name__}. Error: {str(e)}")
        
        # If we get here, all retries failed
        raise RetryExhaustedError(
            f"All {self.max_retries} retries exhausted for {func.__name__}"
        ) from last_exception
    
    def _wait_before_retry(self, attempt: int):
        """Calculate and wait for retry delay"""
        delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        
        if self.jitter:
            delay = random.uniform(0.5 * delay, 1.5 * delay)
        
        print(f"⏳ Waiting {delay:.2f}s before retry {attempt + 1}")
        time.sleep(delay)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator version"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper

class RetryExhaustedError(Exception):
    """Exception raised when all retries are exhausted"""
    pass

# Pre-configured retry mechanisms
class TradingRetryConfig:
    @staticmethod
    def exchange_operation():
        return RetryMechanism(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            jitter=True,
            retry_on_exceptions=(ConnectionError, TimeoutError, Exception)
        )
    
    @staticmethod
    def data_feed_operation():
        return RetryMechanism(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            jitter=True,
            retry_on_exceptions=(ConnectionError, TimeoutError)
        )
    
    @staticmethod
    def api_call():
        return RetryMechanism(
            max_retries=3,
            base_delay=0.5,
            max_delay=5.0,
            jitter=True,
            retry_on_exceptions=(ConnectionError, TimeoutError)
        )

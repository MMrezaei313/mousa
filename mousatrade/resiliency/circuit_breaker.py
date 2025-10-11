import time
import threading
from typing import Callable, Any, Dict
from enum import Enum

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class TradingCircuitBreaker:
    """
    Circuit Breaker pattern for trading operations
    Prevents cascade failures when exchanges are unstable
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 name: str = "default"):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.name = name
        self._lock = threading.RLock()
        self.success_count = 0
        self.half_open_success_threshold = 2
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(str(e))
            raise
    
    def _on_success(self):
        """Handle successful execution"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, error_msg: str):
        """Handle failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state == CircuitState.HALF_OPEN or 
                self.failure_count >= self.failure_threshold):
                self.state = CircuitState.OPEN
                print(f"🚨 Circuit breaker '{self.name}' triggered to OPEN state. Error: {error_msg}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "is_closed": self.state == CircuitState.CLOSED,
            "is_open": self.state == CircuitState.OPEN
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

class CircuitBreakerError(Exception):
    """Custom exception for circuit breaker errors"""
    pass

# Circuit Breaker registry for different services
class CircuitBreakerRegistry:
    _instances = {}
    
    @classmethod
    def get_breaker(cls, name: str, **kwargs) -> TradingCircuitBreaker:
        if name not in cls._instances:
            cls._instances[name] = TradingCircuitBreaker(name=name, **kwargs)
        return cls._instances[name]
    
    @classmethod
    def get_exchange_breaker(cls, exchange_name: str) -> TradingCircuitBreaker:
        return cls.get_breaker(f"exchange_{exchange_name}", failure_threshold=3, recovery_timeout=120)
    
    @classmethod
    def get_api_breaker(cls, service_name: str) -> TradingCircuitBreaker:
        return cls.get_breaker(f"api_{service_name}", failure_threshold=5, recovery_timeout=60)

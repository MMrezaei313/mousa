from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_success_threshold: int = 2

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True

@dataclass
class ResiliencyConfig:
    # Exchange-specific configurations
    exchange_configs: Dict[str, CircuitBreakerConfig] = None
    retry_config: RetryConfig = None
    health_check_interval: int = 30
    
    def __post_init__(self):
        if self.exchange_configs is None:
            self.exchange_configs = {
                "binance": CircuitBreakerConfig(3, 120, 2),
                "robinhood": CircuitBreakerConfig(5, 180, 3),
                "default": CircuitBreakerConfig(5, 60, 2)
            }
        
        if self.retry_config is None:
            self.retry_config = RetryConfig()

# Default configuration
DEFAULT_CONFIG = ResiliencyConfig()

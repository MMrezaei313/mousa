from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import yaml
import os

@dataclass
class CircuitBreakerConfig:
    """Configuration for Circuit Breaker pattern"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_success_threshold: int = 2
    name: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircuitBreakerConfig':
        return cls(**data)

@dataclass
class RetryConfig:
    """Configuration for Retry mechanism"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    retry_on_exceptions: tuple = (Exception,)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetryConfig':
        return cls(**data)

@dataclass
class HealthCheckConfig:
    """Configuration for Health Checks"""
    check_interval: int = 30
    timeout: int = 10
    critical_services: list = None
    
    def __post_init__(self):
        if self.critical_services is None:
            self.critical_services = ["database", "primary_exchange", "data_feed"]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FallbackConfig:
    """Configuration for Fallback strategies"""
    enabled: bool = True
    max_fallback_levels: int = 3
    fallback_timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ResiliencyConfig:
    """
    Main configuration class for all resiliency patterns
    """
    
    # Exchange-specific configurations
    exchange_configs: Dict[str, CircuitBreakerConfig] = None
    retry_config: RetryConfig = None
    health_check_config: HealthCheckConfig = None
    fallback_config: FallbackConfig = None
    
    # Global settings
    enabled: bool = True
    log_level: str = "INFO"
    monitoring_enabled: bool = True
    
    def __post_init__(self):
        if self.exchange_configs is None:
            self.exchange_configs = {
                "binance": CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=120,
                    half_open_success_threshold=2,
                    name="binance"
                ),
                "robinhood": CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=180,
                    half_open_success_threshold=3,
                    name="robinhood"
                ),
                "kucoin": CircuitBreakerConfig(
                    failure_threshold=4,
                    recovery_timeout=90,
                    half_open_success_threshold=2,
                    name="kucoin"
                ),
                "default": CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=60,
                    half_open_success_threshold=2,
                    name="default"
                )
            }
        
        if self.retry_config is None:
            self.retry_config = RetryConfig()
        
        if self.health_check_config is None:
            self.health_check_config = HealthCheckConfig()
        
        if self.fallback_config is None:
            self.fallback_config = FallbackConfig()
    
    def get_exchange_config(self, exchange_name: str) -> CircuitBreakerConfig:
        """Get configuration for specific exchange"""
        return self.exchange_configs.get(
            exchange_name.lower(), 
            self.exchange_configs["default"]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "enabled": self.enabled,
            "log_level": self.log_level,
            "monitoring_enabled": self.monitoring_enabled,
            "exchange_configs": {
                name: config.to_dict() 
                for name, config in self.exchange_configs.items()
            },
            "retry_config": self.retry_config.to_dict(),
            "health_check_config": self.health_check_config.to_dict(),
            "fallback_config": self.fallback_config.to_dict()
        }
    
    def to_yaml(self) -> str:
        """Export configuration to YAML string"""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save_to_file(self, filepath: str):
        """Save configuration to YAML file"""
        with open(filepath, 'w') as f:
            f.write(self.to_yaml())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResiliencyConfig':
        """Create configuration from dictionary"""
        exchange_configs = {}
        if "exchange_configs" in data:
            for name, config_data in data["exchange_configs"].items():
                exchange_configs[name] = CircuitBreakerConfig.from_dict(config_data)
        
        retry_config = RetryConfig.from_dict(data.get("retry_config", {}))
        health_check_config = HealthCheckConfig(**data.get("health_check_config", {}))
        fallback_config = FallbackConfig(**data.get("fallback_config", {}))
        
        return cls(
            exchange_configs=exchange_configs,
            retry_config=retry_config,
            health_check_config=health_check_config,
            fallback_config=fallback_config,
            enabled=data.get("enabled", True),
            log_level=data.get("log_level", "INFO"),
            monitoring_enabled=data.get("monitoring_enabled", True)
        )
    
    @classmethod
    def from_yaml(cls, yaml_string: str) -> 'ResiliencyConfig':
        """Create configuration from YAML string"""
        data = yaml.safe_load(yaml_string)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'ResiliencyConfig':
        """Load configuration from YAML file"""
        if not os.path.exists(filepath):
            return cls()
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data) if data else cls()

# Default configuration instances
DEFAULT_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig()
DEFAULT_RETRY_CONFIG = RetryConfig()
DEFAULT_HEALTH_CHECK_CONFIG = HealthCheckConfig()
DEFAULT_FALLBACK_CONFIG = FallbackConfig()

# Main default configuration
DEFAULT_CONFIG = ResiliencyConfig()

# Environment-specific configurations
PRODUCTION_CONFIG = ResiliencyConfig(
    exchange_configs={
        "binance": CircuitBreakerConfig(failure_threshold=2, recovery_timeout=300),
        "robinhood": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=300),
        "default": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=120)
    },
    retry_config=RetryConfig(max_retries=2, base_delay=2.0),
    health_check_config=HealthCheckConfig(check_interval=15, timeout=5),
    fallback_config=FallbackConfig(max_fallback_levels=2),
    log_level="WARNING"
)

DEVELOPMENT_CONFIG = ResiliencyConfig(
    exchange_configs={
        "default": CircuitBreakerConfig(failure_threshold=10, recovery_timeout=30)
    },
    retry_config=RetryConfig(max_retries=5, base_delay=0.5),
    health_check_config=HealthCheckConfig(check_interval=60),
    log_level="DEBUG"
)

def get_config(environment: str = "default") -> ResiliencyConfig:
    """Get configuration for specific environment"""
    configs = {
        "production": PRODUCTION_CONFIG,
        "dev": DEVELOPMENT_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "default": DEFAULT_CONFIG
    }
    return configs.get(environment.lower(), DEFAULT_CONFIG)

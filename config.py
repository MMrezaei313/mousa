"""
Mousa Trading System - Configuration Module
Enhanced with Multi-Market and Resiliency Settings
"""

import os
import json
from typing import Dict, Any, Optional

class Config:
    """
    Enhanced configuration manager for Mousa Trading System
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config_data = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        default_config = self._get_default_config()
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge with default config
                    return self._deep_merge(default_config, file_config)
            except Exception as e:
                print(f"⚠️ Error loading config file: {e}. Using default config.")
                return default_config
        else:
            print("⚠️ Config file not found. Creating default config.")
            self._save_config(default_config)
            return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "mousa": {
                "version": "2.0.0",
                "environment": "production",
                "log_level": "INFO"
            },
            "exchanges": {
                "binance": {
                    "enabled": True,
                    "api_key": os.getenv("BINANCE_API_KEY", ""),
                    "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                    "sandbox": False,
                    "rate_limit": True
                },
                "robinhood": {
                    "enabled": False,
                    "username": os.getenv("ROBINHOOD_USERNAME", ""),
                    "password": os.getenv("ROBINHOOD_PASSWORD", ""),
                    "mfa_code": os.getenv("ROBINHOOD_MFA", "")
                },
                "forex": {
                    "enabled": False,
                    "broker": "oanda",
                    "api_key": os.getenv("FOREX_API_KEY", ""),
                    "account_id": os.getenv("FOREX_ACCOUNT_ID", ""),
                    "practice_account": True
                },
                "kucoin": {
                    "enabled": False,
                    "api_key": os.getenv("KUCOIN_API_KEY", ""),
                    "api_secret": os.getenv("KUCOIN_API_SECRET", ""),
                    "api_passphrase": os.getenv("KUCOIN_PASSPHRASE", "")
                }
            },
            "trading": {
                "demo_mode": True,
                "max_position_size": 1000,
                "risk_per_trade": 1.0,
                "max_daily_loss": 5.0,
                "default_timeframe": "1h",
                "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            "resiliency": {
                "enabled": True,
                "circuit_breaker_failures": 3,
                "retry_attempts": 3,
                "health_check_interval": 30,
                "fallback_enabled": True
            },
            "monitoring": {
                "arbitrage_check_interval": 30,
                "correlation_check_interval": 300,
                "health_check_interval": 60,
                "portfolio_update_interval": 120
            },
            "notifications": {
                "telegram": {
                    "enabled": True,
                    "token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                    "chat_id": os.getenv("TELEGRAM_CHAT_ID", "")
                },
                "discord": {
                    "enabled": False,
                    "webhook_url": os.getenv("DISCORD_WEBHOOK", "")
                }
            },
            "analysis": {
                "volatility_period": 20,
                "correlation_lookback": 30,
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            },
            "multi_market": {
                "enabled": True,
                "preferred_exchanges": ["binance", "kucoin", "robinhood"],
                "arbitrage_threshold": 0.5,
                "cross_market_trading": True
            }
        }
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _save_config(self, config_data: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"✅ Config saved to {self.config_path}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot notation key"""
        keys = key.split('.')
        config_ref = self.config_data
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
        self._save_config(self.config_data)
    
    def save(self):
        """Save current configuration to file"""
        self._save_config(self.config_data)
    
    def reload(self):
        """Reload configuration from file"""
        self.config_data = self._load_config()
    
    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """Get configuration for specific exchange"""
        return self.get(f"exchanges.{exchange_name}", {})
    
    def get_multi_market_symbols(self) -> Dict[str, list]:
        """Get symbols for multi-market trading"""
        return {
            "crypto": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"],
            "stocks": ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "SPY"],
            "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
        }
    
    def is_exchange_enabled(self, exchange_name: str) -> bool:
        """Check if an exchange is enabled"""
        return self.get(f"exchanges.{exchange_name}.enabled", False)
    
    def get_enabled_exchanges(self) -> list:
        """Get list of enabled exchanges"""
        exchanges = self.get("exchanges", {})
        return [name for name, config in exchanges.items() 
                if config.get("enabled", False)]
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        required_fields = [
            "exchanges.binance.api_key",
            "exchanges.binance.api_secret"
        ]
        
        for field in required_fields:
            if not self.get(field):
                print(f"❌ Missing required config: {field}")
                return False
        
        print("✅ Configuration validated successfully")
        return True

# Global config instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance"""
    return config

def init_config(config_path: str = "config.json") -> Config:
    """Initialize global configuration"""
    global config
    config = Config(config_path)
    return config

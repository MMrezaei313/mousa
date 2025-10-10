import os
import json
import yaml
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

class ConfigLoader:
    def __init__(self, config_path: str = "config"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.logger = self._setup_logging()
        self._load_environment_variables()
        self._load_all_configs()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/config_loader.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_environment_variables(self):
        """Load environment variables from .env file"""
        load_dotenv()
        self.logger.info("Environment variables loaded")
    
    def _load_all_configs(self):
        """Load all configuration files"""
        config_files = {
            'app': 'app_config.json',
            'trading': 'trading_config.json', 
            'api': 'api_config.json',
            'database': 'database_config.json',
            'logging': 'logging_config.json'
        }
        
        for config_type, filename in config_files.items():
            filepath = os.path.join(self.config_path, filename)
            self._load_single_config(config_type, filepath)
        
        self._load_secrets()
        self._validate_config()
    
    def _load_single_config(self, config_type: str, filepath: str):
        """Load a single configuration file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    if filepath.endswith('.json'):
                        self.config[config_type] = json.load(f)
                    elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                        self.config[config_type] = yaml.safe_load(f)
                self.logger.info(f"Loaded {config_type} configuration from {filepath}")
            else:
                self.logger.warning(f"Config file not found: {filepath}")
                self.config[config_type] = self._get_default_config(config_type)
        except Exception as e:
            self.logger.error(f"Error loading {config_type} config: {e}")
            self.config[config_type] = self._get_default_config(config_type)
    
    def _load_secrets(self):
        """Load sensitive data from environment variables"""
        secrets = {
            'binance_api_key': os.getenv('BINANCE_API_KEY', ''),
            'binance_secret_key': os.getenv('BINANCE_SECRET_KEY', ''),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
            'news_api_key': os.getenv('NEWS_API_KEY', ''),
            'database_url': os.getenv('DATABASE_URL', 'sqlite:///data/trading_bot.db')
        }
        
        self.config['secrets'] = secrets
        self.logger.info("Secrets loaded from environment variables")
    
    def _get_default_config(self, config_type: str) -> Dict[str, Any]:
        """Get default configuration for each type"""
        default_configs = {
            'app': {
                'name': 'Mousa Trading Bot',
                'version': '1.0.0',
                'environment': 'development',
                'debug': True,
                'host': '0.0.0.0',
                'port': 5000
            },
            'trading': {
                'enabled': False,
                'demo_mode': True,
                'risk_management': {
                    'max_position_size': 0.1,
                    'stop_loss_percent': 2.0,
                    'take_profit_percent': 4.0,
                    'max_daily_loss': 0.05,
                    'max_portfolio_risk': 0.1
                },
                'strategies': {
                    'technical': {
                        'enabled': True,
                        'timeframes': ['15m', '1h', '4h'],
                        'indicators': ['rsi', 'macd', 'bollinger_bands']
                    },
                    'momentum': {
                        'enabled': True,
                        'lookback_period': 14
                    }
                },
                'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT']
            },
            'api': {
                'binance': {
                    'base_url': 'https://api.binance.com',
                    'testnet_url': 'https://testnet.binance.vision',
                    'timeout': 10,
                    'retries': 3
                },
                'news': {
                    'base_url': 'https://newsapi.org/v2',
                    'timeout': 5
                }
            },
            'database': {
                'default': 'sqlite',
                'sqlite': {
                    'path': 'data/trading_bot.db',
                    'timeout': 30
                },
                'postgresql': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'trading_bot',
                    'pool_size': 10
                }
            },
            'logging': {
                'level': 'INFO',
                'file_path': 'logs/trading_bot.log',
                'max_file_size': 10485760,  # 10MB
                'backup_count': 5,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        return default_configs.get(config_type, {})
    
    def _validate_config(self):
        """Validate the loaded configuration"""
        try:
            # Validate required secrets
            required_secrets = ['binance_api_key', 'binance_secret_key']
            for secret in required_secrets:
                if not self.get(f'secrets.{secret}'):
                    self.logger.warning(f"Required secret {secret} is not set")
            
            # Validate trading configuration
            trading_config = self.get('trading')
            if trading_config:
                risk_config = trading_config.get('risk_management', {})
                if risk_config.get('max_position_size', 0) > 0.5:
                    self.logger.warning("Max position size is too high, consider reducing to 0.5 or less")
            
            self.logger.info("Configuration validation completed")
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        except (KeyError, AttributeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        try:
            keys = key.split('.')
            config = self.config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            self.logger.info(f"Configuration updated: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Error setting configuration {key}: {e}")
    
    def save_config(self, config_type: str, filepath: Optional[str] = None):
        """Save configuration to file"""
        try:
            if filepath is None:
                filepath = os.path.join(self.config_path, f"{config_type}_config.json")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            config_data = self.config.get(config_type, {})
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration to {filepath}: {e}")
            return False
    
    def reload_config(self):
        """Reload all configurations"""
        self.logger.info("Reloading all configurations...")
        self.config.clear()
        self._load_all_configs()
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations (excluding secrets)"""
        safe_config = self.config.copy()
        if 'secrets' in safe_config:
            safe_config['secrets'] = {k: '***' if v else '' for k, v in safe_config['secrets'].items()}
        return safe_config
    
    def create_config_template(self, output_dir: str = "config_templates"):
        """Create configuration template files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            template_configs = {
                'app_config.json': self._get_default_config('app'),
                'trading_config.json': self._get_default_config('trading'),
                'api_config.json': self._get_default_config('api'),
                'database_config.json': self._get_default_config('database'),
                'logging_config.json': self._get_default_config('logging')
            }
            
            for filename, config in template_configs.items():
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.logger.info(f"Template created: {filepath}")
            
            # Create .env template
            env_template = """# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# News API Configuration
NEWS_API_KEY=your_news_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///data/trading_bot.db
"""
            
            env_path = os.path.join(output_dir, '.env.template')
            with open(env_path, 'w') as f:
                f.write(env_template)
            
            self.logger.info(f"Environment template created: {env_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating config templates: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize config loader
    config_loader = ConfigLoader()
    
    # Get configuration values
    app_name = config_loader.get('app.name')
    trading_enabled = config_loader.get('trading.enabled')
    binance_api_key = config_loader.get('secrets.binance_api_key')
    
    print(f"App Name: {app_name}")
    print(f"Trading Enabled: {trading_enabled}")
    print(f"Binance API Key: {'***' if binance_api_key else 'Not set'}")
    
    # Set configuration value
    config_loader.set('trading.demo_mode', True)
    
    # Save configuration
    config_loader.save_config('trading')
    
    # Get all configurations (safe version)
    all_configs = config_loader.get_all_configs()
    print("All configurations:", json.dumps(all_configs, indent=2))
    
    # Create config templates
    config_loader.create_config_template()

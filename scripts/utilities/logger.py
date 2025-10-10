import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json

class TradingLogger:
    def __init__(self, name: str = "trading_bot", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging configuration"""
        
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Create logs directory
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.get('console_level', 'INFO'))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'trading_bot.log'),
            maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),  # 10MB
            backupCount=self.config.get('backup_count', 5)
        )
        file_handler.setLevel(self.config.get('file_level', 'DEBUG'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Error handler for error logs only
        error_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'errors.log'),
            maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
            backupCount=self.config.get('backup_count', 5)
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # Trading signals handler
        signal_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'trading_signals.log'),
            maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
            backupCount=self.config.get('backup_count', 5)
        )
        signal_handler.setLevel(logging.INFO)
        signal_formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        signal_handler.setFormatter(signal_formatter)
        signal_handler.addFilter(TradingSignalFilter())
        logger.addHandler(signal_handler)
        
        return logger
    
    def info(self, message: str, extra_data: Optional[Dict] = None):
        """Log info message with optional extra data"""
        if extra_data:
            message = f"{message} | {json.dumps(extra_data)}"
        self.logger.info(message)
    
    def error(self, message: str, exception: Optional[Exception] = None, extra_data: Optional[Dict] = None):
        """Log error message with exception and extra data"""
        if exception:
            message = f"{message} - Exception: {str(exception)}"
        if extra_data:
            message = f"{message} | {json.dumps(extra_data)}"
        self.logger.error(message, exc_info=exception is not None)
    
    def warning(self, message: str, extra_data: Optional[Dict] = None):
        """Log warning message with extra data"""
        if extra_data:
            message = f"{message} | {json.dumps(extra_data)}"
        self.logger.warning(message)
    
    def debug(self, message: str, extra_data: Optional[Dict] = None):
        """Log debug message with extra data"""
        if extra_data:
            message = f"{message} | {json.dumps(extra_data)}"
        self.logger.debug(message)
    
    def trading_signal(self, symbol: str, signal: str, price: float, strength: float, reasons: str):
        """Log trading signal in specialized format"""
        message = f"SIGNAL | {symbol} | {signal} | Price: {price:.2f} | Strength: {strength:.2f} | Reasons: {reasons}"
        self.logger.info(message)
    
    def trade_execution(self, symbol: str, action: str, quantity: float, price: float, pnl: float = 0):
        """Log trade execution"""
        message = f"TRADE | {symbol} | {action} | Qty: {quantity:.4f} | Price: {price:.2f} | PnL: {pnl:.2f}"
        self.logger.info(message)
    
    def performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        message = f"PERFORMANCE | {json.dumps(metrics)}"
        self.logger.info(message)
    
    def market_data(self, symbol: str, data_type: str, details: str):
        """Log market data events"""
        message = f"MARKET_DATA | {symbol} | {data_type} | {details}"
        self.logger.debug(message)
    
    def system_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """Log system alerts"""
        formatted_message = f"SYSTEM_ALERT | {alert_type} | {severity} | {message}"
        
        if severity == "ERROR":
            self.logger.error(formatted_message)
        elif severity == "WARNING":
            self.logger.warning(formatted_message)
        else:
            self.logger.info(formatted_message)
    
    def get_log_file_paths(self) -> Dict[str, str]:
        """Get paths to all log files"""
        log_dir = self.config.get('log_dir', 'logs')
        return {
            'main_log': os.path.join(log_dir, 'trading_bot.log'),
            'error_log': os.path.join(log_dir, 'errors.log'),
            'signal_log': os.path.join(log_dir, 'trading_signals.log')
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days"""
        try:
            log_dir = self.config.get('log_dir', 'logs')
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            deleted_files = []
            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.isfile(filepath):
                    file_time = os.path.getmtime(filepath)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        deleted_files.append(filename)
            
            if deleted_files:
                self.info(f"Cleaned up {len(deleted_files)} old log files", {'deleted_files': deleted_files})
            else:
                self.debug("No old log files to clean up")
                
            return deleted_files
            
        except Exception as e:
            self.error("Error cleaning up old logs", e)
            return []

class TradingSignalFilter(logging.Filter):
    """Filter for trading signal logs"""
    
    def filter(self, record):
        return 'SIGNAL' in record.getMessage() or 'TRADE' in record.getMessage()

class DatabaseHandler(logging.Handler):
    """Custom handler for logging to database"""
    
    def __init__(self, db_connection, table_name='system_logs'):
        super().__init__()
        self.db_connection = db_connection
        self.table_name = table_name
        self._create_table()
    
    def _create_table(self):
        """Create logs table if not exists"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    logger_name TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    function TEXT,
                    line_number INTEGER,
                    extra_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.db_connection.commit()
        except Exception as e:
            print(f"Error creating log table: {e}")
    
    def emit(self, record):
        """Emit log record to database"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute(f'''
                INSERT INTO {self.table_name} 
                (timestamp, logger_name, level, message, module, function, line_number, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.fromtimestamp(record.created),
                record.name,
                record.levelname,
                record.getMessage(),
                record.module,
                record.funcName,
                record.lineno,
                getattr(record, 'extra_data', None)
            ))
            
            self.db_connection.commit()
        except Exception as e:
            print(f"Error writing log to database: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize logger with configuration
    logger_config = {
        'log_dir': 'logs',
        'console_level': 'INFO',
        'file_level': 'DEBUG',
        'max_file_size': 5 * 1024 * 1024,  # 5MB
        'backup_count': 3
    }
    
    trading_logger = TradingLogger('mousa_trading_bot', logger_config)
    
    # Test different log types
    trading_logger.info("Application started successfully")
    
    trading_logger.trading_signal(
        symbol='BTCUSDT',
        signal='BUY',
        price=50123.45,
        strength=0.85,
        reasons='RSI oversold, MACD bullish crossover'
    )
    
    trading_logger.trade_execution(
        symbol='ETHUSDT',
        action='SELL',
        quantity=0.5,
        price=3250.67,
        pnl=125.50
    )
    
    trading_logger.performance_metrics({
        'total_return': 15.5,
        'sharpe_ratio': 1.8,
        'win_rate': 65.2,
        'max_drawdown': -8.7
    })
    
    trading_logger.system_alert(
        alert_type='API_CONNECTION',
        message='Binance API connection restored',
        severity='INFO'
    )
    
    # Log with extra data
    trading_logger.info(
        "Market data update completed",
        extra_data={'symbols_updated': 15, 'time_taken': 2.5}
    )
    
    # Get log file paths
    log_paths = trading_logger.get_log_file_paths()
    print("Log files:", log_paths)
    
    # Cleanup old logs (keep last 30 days)
    # trading_logger.cleanup_old_logs(30)

import logging
import sys
from typing import Dict, Any, Optional, Callable
from functools import wraps

class ErrorHandler:
    """
    Centralized error handling for trading operations
    """
    
    def __init__(self, log_file: str = "trading_errors.log"):
        self.logger = self._setup_logger(log_file)
        self.error_counts = {}
        
    def _setup_logger(self, log_file: str) -> logging.Logger:
        """Setup error logger"""
        logger = logging.getLogger("TradingErrorHandler")
        logger.setLevel(logging.ERROR)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def handle_trading_error(self, error: Exception, context: Dict[str, Any] = None):
        """Handle trading-related errors"""
        error_name = type(error).__name__
        
        # Update error counts
        self.error_counts[error_name] = self.error_counts.get(error_name, 0) + 1
        
        # Log error with context
        log_message = f"Trading Error: {error_name} - {str(error)}"
        if context:
            log_message += f" | Context: {context}"
        
        self.logger.error(log_message)
        
        # Specific handling for different error types
        if "Connection" in error_name:
            self._handle_connection_error(error, context)
        elif "Timeout" in error_name:
            self._handle_timeout_error(error, context)
        elif "InsufficientFunds" in error_name:
            self._handle_insufficient_funds(error, context)
        else:
            self._handle_generic_error(error, context)
    
    def _handle_connection_error(self, error: Exception, context: Dict[str, Any]):
        """Handle connection-related errors"""
        print("🔌 Connection error detected. Check network connectivity.")
        
    def _handle_timeout_error(self, error: Exception, context: Dict[str, Any]):
        """Handle timeout errors"""
        print("⏰ Timeout error. Consider increasing timeout thresholds.")
        
    def _handle_insufficient_funds(self, error: Exception, context: Dict[str, Any]):
        """Handle insufficient funds errors"""
        print("💰 Insufficient funds. Check account balance.")
        
    def _handle_generic_error(self, error: Exception, context: Dict[str, Any]):
        """Handle generic errors"""
        print("❌ Generic error occurred. Check logs for details.")
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return self.error_counts.copy()
    
    def reset_error_counts(self):
        """Reset error counters"""
        self.error_counts.clear()

# Decorator for automatic error handling
def catch_trading_errors(context: Dict[str, Any] = None):
    """Decorator for automatic error handling in trading functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or {}
                error_context.update({
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
                error_handler.handle_trading_error(e, error_context)
                raise
        return wrapper
    return decorator

import os
from datetime import datetime

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Trading APIs (you'll need to add your actual keys)
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')
    
    # Trading parameters
    DEFAULT_TIMEFRAME = '1h'
    SUPPORTED_PAIRS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    # Risk management
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio
    STOP_LOSS_PERCENT = 2.0
    TAKE_PROFIT_PERCENT = 4.0

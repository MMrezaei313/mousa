"""
Pytest configuration and fixtures for Mousa Trading Bot tests
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1h')
    
    # Generate realistic price data with some trends and volatility
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 1000))),
        'close': prices,
        'volume': np.random.normal(1000, 200, 1000)
    }, index=dates)
    
    return data

@pytest.fixture
def sample_strategy_parameters():
    """Sample strategy parameters for testing"""
    return {
        'moving_average': {
            'fast_window': 10,
            'slow_window': 30
        },
        'rsi': {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70
        },
        'macd': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    }

@pytest.fixture
def temp_database():
    """Create a temporary database for testing"""
    import tempfile
    import sqlite3
    
    # Create temporary database file
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    # Create basic schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp DATETIME,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            symbol TEXT,
            signal TEXT,
            strength REAL,
            timestamp DATETIME
        )
    ''')
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

@pytest.fixture
def sample_optimization_parameters():
    """Sample optimization parameters for testing"""
    return {
        'fast_window': (5, 50, 'int'),
        'slow_window': (20, 200, 'int'),
        'rsi_period': (10, 30, 'int')
    }

@pytest.fixture
def mock_web_client():
    """Mock web client for testing web endpoints"""
    from unittest.mock import MagicMock
    
    client = MagicMock()
    client.get.return_value.status_code = 200
    client.post.return_value.status_code = 200
    client.put.return_value.status_code = 200
    client.delete.return_value.status_code = 200
    
    return client

@pytest.fixture
def sample_portfolio_data():
    """Generate sample portfolio data for testing"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
    
    portfolio = {}
    for symbol in symbols:
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.03, 100)
        portfolio[symbol] = pd.Series(returns, index=dates)
    
    return portfolio

@pytest.fixture
def sample_ml_features():
    """Generate sample ML features for testing"""
    dates = pd.date_range(start='2023-01-01', periods=500, freq='1h')
    
    features = pd.DataFrame({
        'returns': np.random.normal(0, 0.01, 500),
        'volatility': np.random.uniform(0.01, 0.05, 500),
        'rsi': np.random.uniform(20, 80, 500),
        'macd': np.random.normal(0, 0.01, 500),
        'volume_ratio': np.random.uniform(0.5, 2.0, 500)
    }, index=dates)
    
    # Create target variable (1 if next return is positive, 0 otherwise)
    features['target'] = (features['returns'].shift(-1) > 0).astype(int)
    
    return features.dropna()

@pytest.fixture
def sample_backtest_results():
    """Sample backtest results for testing"""
    return {
        'total_return': 0.156,
        'annual_return': 0.234,
        'sharpe_ratio': 1.89,
        'max_drawdown': -0.089,
        'win_rate': 0.634,
        'total_trades': 147,
        'equity_curve': pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100))),
        'trades': [
            {'entry_time': '2023-01-01', 'exit_time': '2023-01-02', 'pnl': 250},
            {'entry_time': '2023-01-03', 'exit_time': '2023-01-04', 'pnl': -120}
        ]
    }

def pytest_configure(config):
    """Pytest configuration hook"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "web: mark test as web-related"
    )

def pytest_runtest_setup(item):
    """Setup before each test"""
    # Skip slow tests unless explicitly requested
    if 'slow' in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("slow test: use --run-slow to run")

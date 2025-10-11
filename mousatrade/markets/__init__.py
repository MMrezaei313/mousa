"""
Multi-Market Support for Mousa Trading System
Support for Crypto, Forex, Stocks, Futures, and Indices
"""

from .base_market import BaseMarket, MarketType, TradingSession
from .market_registry import MarketRegistry
from .data_feeds import MultiTimeframeDataFeed, BaseDataFeed

# Crypto markets
from .crypto import (
    BinanceMarket,
    KuCoinMarket,
    CryptoAnalyzer
)

# Stock markets  
from .stocks import (
    RobinhoodMarket,
    AlpacaMarket,
    StockAnalyzer
)

# Forex markets
from .forex import (
    ForexMarket,
    ForexAnalyzer
)

# Futures markets
from .futures import (
    FuturesMarket,
    FuturesAnalyzer
)

# Indices markets
from .indices import (
    IndicesMarket,
    IndicesAnalyzer
)

# Utilities
from .utils import (
    VolatilityCalculator,
    CorrelationTracker,
    MarketHoursManager,
    SymbolNormalizer
)

__version__ = "1.0.0"
__author__ = "Mousa Trading Team"

__all__ = [
    # Base classes
    'BaseMarket',
    'MarketType', 
    'TradingSession',
    'MarketRegistry',
    
    # Data feeds
    'BaseDataFeed',
    'MultiTimeframeDataFeed',
    
    # Crypto
    'BinanceMarket',
    'KuCoinMarket', 
    'CryptoAnalyzer',
    
    # Stocks
    'RobinhoodMarket',
    'AlpacaMarket',
    'StockAnalyzer',
    
    # Forex
    'ForexMarket',
    'ForexAnalyzer',
    
    # Futures
    'FuturesMarket', 
    'FuturesAnalyzer',
    
    # Indices
    'IndicesMarket',
    'IndicesAnalyzer',
    
    # Utilities
    'VolatilityCalculator',
    'CorrelationTracker',
    'MarketHoursManager',
    'SymbolNormalizer'
]

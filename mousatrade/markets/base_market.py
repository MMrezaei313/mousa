from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, time

class MarketType(Enum):
    CRYPTO = "crypto"
    FOREX = "forex" 
    STOCKS = "stocks"
    FUTURES = "futures"
    INDICES = "indices"

class TradingSession(Enum):
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    _24_7 = "24_7"  # For crypto markets

@dataclass
class MarketHours:
    open_time: time
    close_time: time 
    timezone: str
    sessions: Dict[TradingSession, tuple] = None

@dataclass
class SymbolInfo:
    symbol: str
    name: str
    market_type: MarketType
    base_currency: str
    quote_currency: str
    min_trade_size: float
    price_precision: int
    quantity_precision: int
    is_active: bool = True

class BaseMarket(ABC):
    """
    Abstract base class for all market implementations
    """
    
    def __init__(self, market_type: MarketType, name: str):
        self.market_type = market_type
        self.name = name
        self.connected = False
        self.symbols_info: Dict[str, SymbolInfo] = {}
        self.market_hours: Optional[MarketHours] = None
        
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """Connect to market data and trading"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from market"""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Get account balances"""
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker price"""
        pass
    
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data"""
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, order_type: str, side: str, 
                   quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a new order"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        pass
    
    # Common functionality
    def get_symbols(self) -> List[str]:
        """Get list of available symbols"""
        return list(self.symbols_info.keys())
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get information for a specific symbol"""
        return self.symbols_info.get(symbol.upper())
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        if self.market_type == MarketType.CRYPTO:
            return True  # Crypto markets are always open
        
        if not self.market_hours:
            return False
            
        now = datetime.now().time()
        return self.market_hours.open_time <= now <= self.market_hours.close_time
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format"""
        return symbol.upper().replace('/', '').replace('-', '')
    
    def calculate_position_size(self, symbol: str, risk_percent: float, 
                              stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return 0.0
            
        balance = self.get_balance()
        account_balance = balance.get(symbol_info.quote_currency, 0)
        
        risk_amount = account_balance * (risk_percent / 100)
        position_size = risk_amount / abs(stop_loss)
        
        return min(position_size, account_balance * 0.1)  # Max 10% of account

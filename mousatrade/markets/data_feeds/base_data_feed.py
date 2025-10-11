from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable
import pandas as pd
from ..base_market import MarketType

class BaseDataFeed(ABC):
    """
    Abstract base class for market data feeds
    """
    
    def __init__(self, market_type: MarketType, name: str):
        self.market_type = market_type
        self.name = name
        self.connected = False
        self.subscribers: Dict[str, List[Callable]] = {}
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to data feed"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from data feed"""
        pass
    
    @abstractmethod
    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to real-time data for symbol"""
        pass
    
    @abstractmethod
    def unsubscribe(self, symbol: str, callback: Callable = None):
        """Unsubscribe from symbol"""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical OHLCV data"""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        pass
    
    def notify_subscribers(self, symbol: str, data: Dict):
        """Notify all subscribers of new data"""
        if symbol in self.subscribers:
            for callback in self.subscribers[symbol]:
                try:
                    callback(symbol, data)
                except Exception as e:
                    print(f"Error in data feed callback: {e}")

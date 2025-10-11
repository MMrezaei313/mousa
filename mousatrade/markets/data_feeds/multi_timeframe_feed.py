import pandas as pd
from typing import Dict, List, Optional, Callable
from threading import Lock
from .base_data_feed import BaseDataFeed
from ..base_market import MarketType

class MultiTimeframeDataFeed(BaseDataFeed):
    """
    Data feed that provides multiple timeframe analysis
    """
    
    def __init__(self, base_feed: BaseDataFeed, timeframes: List[str]):
        super().__init__(base_feed.market_type, f"MultiTF_{base_feed.name}")
        self.base_feed = base_feed
        self.timeframes = timeframes
        self.data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._lock = Lock()
        
        # Subscribe to base feed updates
        self.base_feed.subscribe('*', self._on_base_data)
    
    def _on_base_data(self, symbol: str, data: Dict):
        """Handle new data from base feed"""
        with self._lock:
            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            
            # Update all timeframes with new data
            for timeframe in self.timeframes:
                self._update_timeframe(symbol, timeframe, data)
            
            # Notify subscribers
            self.notify_subscribers(symbol, {
                'price': data.get('price'),
                'timeframes': self.data_cache[symbol],
                'timestamp': data.get('timestamp')
            })
    
    def _update_timeframe(self, symbol: str, timeframe: str, new_data: Dict):
        """Update specific timeframe data"""
        # This is a simplified implementation
        # In production, you'd use proper OHLCV aggregation
        
        if timeframe not in self.data_cache[symbol]:
            self.data_cache[symbol][timeframe] = pd.DataFrame()
        
        # Add new data point (implementation depends on your needs)
        pass
    
    def connect(self) -> bool:
        return self.base_feed.connect()
    
    def disconnect(self):
        self.base_feed.disconnect()
    
    def subscribe(self, symbol: str, callback: Callable):
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
    
    def unsubscribe(self, symbol: str, callback: Callable = None):
        if symbol in self.subscribers:
            if callback:
                self.subscribers[symbol].remove(callback)
            else:
                del self.subscribers[symbol]
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        return self.base_feed.get_historical_data(symbol, timeframe, start_date, end_date)
    
    def get_current_price(self, symbol: str) -> float:
        return self.base_feed.get_current_price(symbol)
    
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get data for all timeframes for a symbol"""
        with self._lock:
            return self.data_cache.get(symbol, {}).copy()

from typing import Dict, List, Optional, Any
from threading import Lock
from .base_market import BaseMarket, MarketType

class MarketRegistry:
    """
    Central registry for managing multiple markets
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.markets: Dict[str, BaseMarket] = {}
            self.market_connections: Dict[str, bool] = {}
            self._initialized = True
    
    def register_market(self, name: str, market: BaseMarket):
        """Register a new market"""
        self.markets[name] = market
        self.market_connections[name] = False
    
    def unregister_market(self, name: str):
        """Unregister a market"""
        if name in self.markets:
            if self.market_connections.get(name, False):
                self.markets[name].disconnect()
            del self.markets[name]
            del self.market_connections[name]
    
    def connect_all(self, **kwargs) -> Dict[str, bool]:
        """Connect to all registered markets"""
        results = {}
        for name, market in self.markets.items():
            try:
                connected = market.connect(**kwargs)
                self.market_connections[name] = connected
                results[name] = connected
            except Exception as e:
                results[name] = False
                print(f"Failed to connect {name}: {e}")
        return results
    
    def disconnect_all(self):
        """Disconnect from all markets"""
        for name, market in self.markets.items():
            try:
                market.disconnect()
                self.market_connections[name] = False
            except Exception as e:
                print(f"Error disconnecting {name}: {e}")
    
    def get_market(self, name: str) -> Optional[BaseMarket]:
        """Get a specific market by name"""
        return self.markets.get(name)
    
    def get_connected_markets(self) -> List[str]:
        """Get list of connected markets"""
        return [name for name, connected in self.market_connections.items() if connected]
    
    def get_markets_by_type(self, market_type: MarketType) -> List[BaseMarket]:
        """Get all markets of a specific type"""
        return [market for market in self.markets.values() 
                if market.market_type == market_type]
    
    def get_balances(self) -> Dict[str, Dict[str, float]]:
        """Get balances from all connected markets"""
        balances = {}
        for name, market in self.markets.items():
            if self.market_connections.get(name, False):
                try:
                    balances[name] = market.get_balance()
                except Exception as e:
                    print(f"Error getting balance from {name}: {e}")
                    balances[name] = {}
        return balances
    
    def execute_cross_market_order(self, symbol: str, order_type: str, side: str,
                                 quantity: float, preferred_markets: List[str] = None):
        """
        Execute order across multiple markets with fallback
        """
        markets_to_try = preferred_markets or list(self.markets.keys())
        
        for market_name in markets_to_try:
            market = self.get_market(market_name)
            if not market or not self.market_connections.get(market_name, False):
                continue
                
            try:
                # Check if symbol is available in this market
                if symbol not in market.get_symbols():
                    continue
                
                result = market.place_order(symbol, order_type, side, quantity)
                return {
                    "success": True,
                    "market": market_name,
                    "result": result
                }
                
            except Exception as e:
                print(f"Order failed on {market_name}: {e}")
                continue
        
        return {
            "success": False,
            "error": "Order failed on all markets"
        }
    
    def get_best_price(self, symbol: str, side: str) -> Dict[str, Any]:
        """
        Get best price across all markets for a symbol
        """
        best_price = None
        best_market = None
        
        for name, market in self.markets.items():
            if not self.market_connections.get(name, False):
                continue
                
            try:
                ticker = market.get_ticker(symbol)
                if not ticker:
                    continue
                    
                price = ticker.get('bid' if side == 'sell' else 'ask')
                if price and (best_price is None or 
                            (price > best_price if side == 'sell' else price < best_price)):
                    best_price = price
                    best_market = name
                    
            except Exception as e:
                print(f"Error getting price from {name}: {e}")
                continue
        
        return {
            "market": best_market,
            "price": best_price,
            "symbol": symbol,
            "side": side
        }

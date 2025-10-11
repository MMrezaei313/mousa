import re
from typing import Dict, List, Optional
from ..base_market import MarketType

class SymbolNormalizer:
    """
    Normalize and convert symbols across different markets and formats
    """
    
    def __init__(self):
        self.symbol_patterns = {
            MarketType.CRYPTO: r'^[A-Z0-9]+/[A-Z0-9]+$',
            MarketType.STOCKS: r'^[A-Z]{1,5}$',
            MarketType.FOREX: r'^[A-Z]{3}/[A-Z]{3}$',
            MarketType.FUTURES: r'^[A-Z]{1,2}[A-Z0-9]{1,3}\d{2}$'
        }
        
        self.symbol_mappings = {
            'BTC/USDT': ['BTCUSDT', 'XBTUSDT', 'BTC-USD'],
            'ETH/USDT': ['ETHUSDT', 'ETH-USD'],
            'AAPL': ['AAPL.US', 'AAPL-US'],
            'SPY': ['SPY.US', 'SPY-US'],
            'EUR/USD': ['EURUSD', 'EUR-USD'],
        }
    
    def normalize_symbol(self, symbol: str, market_type: MarketType) -> str:
        """Normalize symbol to standard format"""
        symbol = symbol.upper().strip()
        
        if market_type == MarketType.CRYPTO:
            # Convert BTCUSDT to BTC/USDT
            if '/' not in symbol and len(symbol) > 6:
                for base in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']:
                    if symbol.startswith(base):
                        quote = symbol[len(base):]
                        if quote in ['USDT', 'USD', 'BUSD']:
                            return f"{base}/{quote}"
        
        elif market_type == MarketType.FOREX:
            # Convert EURUSD to EUR/USD
            if '/' not in symbol and len(symbol) == 6:
                return f"{symbol[:3]}/{symbol[3:]}"
        
        elif market_type == MarketType.STOCKS:
            # Remove suffixes like .US, -US
            symbol = re.sub(r'[\.\-].*$', '', symbol)
        
        return symbol
    
    def convert_symbol(self, symbol: str, from_market: MarketType, 
                      to_market: MarketType) -> Optional[str]:
        """Convert symbol from one market format to another"""
        normalized = self.normalize_symbol(symbol, from_market)
        
        # Simple conversion logic
        if from_market == MarketType.CRYPTO and to_market == MarketType.STOCKS:
            # Crypto to stock-like symbol (for analysis)
            return normalized.replace('/', '')
        
        elif from_market == MarketType.STOCKS and to_market == MarketType.CRYPTO:
            # Stock to crypto-like symbol
            if normalized == 'BTC':  # Example mapping
                return 'BTC/USDT'
        
        return normalized
    
    def validate_symbol(self, symbol: str, market_type: MarketType) -> bool:
        """Validate if symbol matches expected pattern for market"""
        pattern = self.symbol_patterns.get(market_type)
        if not pattern:
            return False
        
        normalized = self.normalize_symbol(symbol, market_type)
        return bool(re.match(pattern, normalized))
    
    def get_equivalent_symbols(self, symbol: str, market_type: MarketType) -> List[str]:
        """Get equivalent symbols across different formats/exchanges"""
        normalized = self.normalize_symbol(symbol, market_type)
        
        equivalents = [normalized]
        
        # Add mappings from predefined dictionary
        for standard, variants in self.symbol_mappings.items():
            if normalized == standard:
                equivalents.extend(variants)
            elif normalized in variants:
                equivalents.append(standard)
                equivalents.extend([v for v in variants if v != normalized])
        
        return list(set(equivalents))
    
    def detect_market_type(self, symbol: str) -> Optional[MarketType]:
        """Detect market type from symbol format"""
        normalized = symbol.upper().strip()
        
        for market_type, pattern in self.symbol_patterns.items():
            if re.match(pattern, normalized):
                return market_type
        
        # Fallback detection
        if '/' in normalized:
            if len(normalized) == 7 and normalized.index('/') == 3:
                return MarketType.FOREX
            else:
                return MarketType.CRYPTO
        elif len(normalized) <= 5 and normalized.isalpha():
            return MarketType.STOCKS
        elif re.match(r'^[A-Z]{1,2}\d{1,2}$', normalized):
            return MarketType.FUTURES
        
        return None

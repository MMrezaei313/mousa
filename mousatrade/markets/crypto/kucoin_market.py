import ccxt
import pandas as pd
from typing import Dict, List, Optional, Any
from ...base_market import BaseMarket, MarketType, SymbolInfo

class KuCoinMarket(BaseMarket):
    """
    KuCoin cryptocurrency market implementation
    """
    
    def __init__(self, api_key: str = None, secret: str = None, password: str = None):
        super().__init__(MarketType.CRYPTO, "KuCoin")
        self.api_key = api_key
        self.secret = secret
        self.password = password
        self.exchange = None
        self._load_symbols_info()
    
    def _load_symbols_info(self):
        """Load available symbols information"""
        kucoin_pairs = [
            ('BTC/USDT', 'Bitcoin', 'BTC', 'USDT', 0.0001, 2, 6),
            ('ETH/USDT', 'Ethereum', 'ETH', 'USDT', 0.001, 2, 4),
            ('KCS/USDT', 'KuCoin Token', 'KCS', 'USDT', 0.1, 4, 1),
            ('DOT/USDT', 'Polkadot', 'DOT', 'USDT', 0.1, 2, 1),
            ('LINK/USDT', 'Chainlink', 'LINK', 'USDT', 0.1, 2, 1),
            ('ADA/USDT', 'Cardano', 'ADA', 'USDT', 1, 4, 0),
        ]
        
        for symbol, name, base, quote, min_size, price_prec, qty_prec in kucoin_pairs:
            self.symbols_info[symbol] = SymbolInfo(
                symbol=symbol,
                name=name,
                market_type=MarketType.CRYPTO,
                base_currency=base,
                quote_currency=quote,
                min_trade_size=min_size,
                price_precision=price_prec,
                quantity_precision=qty_prec
            )
    
    def connect(self, **kwargs) -> bool:
        """Connect to KuCoin"""
        try:
            self.exchange = ccxt.kucoin({
                'apiKey': self.api_key or kwargs.get('api_key'),
                'secret': self.secret or kwargs.get('secret'),
                'password': self.password or kwargs.get('password'),
                'sandbox': kwargs.get('sandbox', False),
                'enableRateLimit': True,
            })
            
            # Test connection
            self.exchange.fetch_balance()
            self.connected = True
            print("✅ Connected to KuCoin")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to KuCoin: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from KuCoin"""
        if self.exchange:
            self.exchange.close()
        self.connected = False
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balances"""
        if not self.connected:
            return {}
        
        try:
            balance = self.exchange.fetch_balance()
            free_balance = {}
            
            for currency, amount in balance['free'].items():
                if amount > 0:
                    free_balance[currency] = float(amount)
            
            return free_balance
            
        except Exception as e:
            print(f"Error getting balance: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker price"""
        if not self.connected:
            return {}
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['baseVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            print(f"Error getting ticker for {symbol}: {e}")
            return {}
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data"""
        if not self.connected:
            return pd.DataFrame()
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            print(f"Error getting OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, order_type: str, side: str, 
                   quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a new order"""
        if not self.connected:
            return {'error': 'Not connected to exchange'}
        
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return {'error': f'Symbol {symbol} not found'}
            
            quantity = round(quantity, symbol_info.quantity_precision)
            
            if order_type.upper() == 'MARKET':
                order = self.exchange.create_market_order(symbol, side.lower(), quantity)
            else:
                if price is None:
                    return {'error': 'Price required for limit order'}
                price = round(price, symbol_info.price_precision)
                order = self.exchange.create_limit_order(symbol, side.lower(), quantity, price)
            
            return {
                'id': order['id'],
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': 'open'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.connected:
            return False
        
        try:
            self.exchange.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if not self.connected:
            return {}
        
        try:
            return self.exchange.fetch_order(order_id)
        except Exception as e:
            print(f"Error getting order {order_id}: {e}")
            return {}

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, time
from ...base_market import BaseMarket, MarketType, SymbolInfo, MarketHours, TradingSession

class AlpacaMarket(BaseMarket):
    """
    Alpaca Markets implementation for US stocks
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        super().__init__(MarketType.STOCKS, "Alpaca")
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.client = None
        self._setup_market_hours()
        self._load_symbols_info()
    
    def _setup_market_hours(self):
        """Setup US stock market hours"""
        self.market_hours = MarketHours(
            open_time=time(9, 30),
            close_time=time(16, 0),
            timezone="America/New_York",
            sessions={
                TradingSession.PRE_MARKET: (time(4, 0), time(9, 30)),
                TradingSession.REGULAR: (time(9, 30), time(16, 0)),
                TradingSession.AFTER_HOURS: (time(16, 0), time(20, 0))
            }
        )
    
    def _load_symbols_info(self):
        """Load Alpaca available symbols"""
        alpaca_symbols = [
            ('SPY', 'SPDR S&P 500 ETF Trust', 'USD', 1, 2, 0),
            ('QQQ', 'Invesco QQQ Trust', 'USD', 1, 2, 0),
            ('IWM', 'iShares Russell 2000 ETF', 'USD', 1, 2, 0),
            ('DIA', 'SPDR Dow Jones Industrial Average ETF', 'USD', 1, 2, 0),
            ('VOO', 'Vanguard S&P 500 ETF', 'USD', 1, 2, 0),
            ('AAPL', 'Apple Inc.', 'USD', 1, 2, 0),
            ('MSFT', 'Microsoft Corporation', 'USD', 1, 2, 0),
            ('GOOGL', 'Alphabet Inc.', 'USD', 1, 2, 0),
            ('AMZN', 'Amazon.com Inc.', 'USD', 1, 2, 0),
            ('TSLA', 'Tesla Inc.', 'USD', 1, 2, 0),
        ]
        
        for symbol, name, currency, min_shares, price_prec, qty_prec in alpaca_symbols:
            self.symbols_info[symbol] = SymbolInfo(
                symbol=symbol,
                name=name,
                market_type=MarketType.STOCKS,
                base_currency=symbol,
                quote_currency=currency,
                min_trade_size=min_shares,
                price_precision=price_prec,
                quantity_precision=qty_prec
            )
    
    def connect(self, **kwargs) -> bool:
        """Connect to Alpaca"""
        try:
            # In production, use: import alpaca_trade_api as tradeapi
            self.api_key = self.api_key or kwargs.get('api_key')
            self.secret_key = self.secret_key or kwargs.get('secret_key')
            self.paper = kwargs.get('paper', self.paper)
            
            # Simulate connection
            # self.client = tradeapi.REST(self.api_key, self.secret_key, base_url='https://paper-api.alpaca.markets' if self.paper else 'https://api.alpaca.markets')
            
            print("✅ Connected to Alpaca (simulated)")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Alpaca: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Alpaca"""
        self.connected = False
        print("🔌 Disconnected from Alpaca")
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balances"""
        if not self.connected:
            return {}
        
        try:
            # Simulated balance - in production, use: self.client.get_account()
            return {
                'USD': 50000.0,
                'SPY': 25.0,
                'AAPL': 50.0
            }
        except Exception as e:
            print(f"Error getting balance: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker price"""
        if not self.connected:
            return {}
        
        try:
            # Simulated data - in production, use: self.client.get_last_trade(symbol)
            import random
            base_prices = {
                'SPY': 450.0, 'QQQ': 370.0, 'IWM': 190.0, 'DIA': 340.0,
                'VOO': 420.0, 'AAPL': 180.0, 'MSFT': 330.0, 'GOOGL': 135.0,
                'AMZN': 145.0, 'TSLA': 240.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            variation = random.uniform(-0.015, 0.015)
            current_price = base_price * (1 + variation)
            
            return {
                'symbol': symbol,
                'bid': current_price * 0.9995,
                'ask': current_price * 1.0005,
                'last': current_price,
                'volume': random.randint(5000000, 15000000),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting ticker for {symbol}: {e}")
            return {}
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1D', limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data"""
        if not self.connected:
            return pd.DataFrame()
        
        try:
            # Simulated data - in production, use: self.client.get_barset(symbol, timeframe, limit=limit)
            import random
            from datetime import datetime, timedelta
            
            base_price = 100.0
            data = []
            current_time = datetime.now()
            
            for i in range(limit):
                open_price = base_price * random.uniform(0.97, 1.03)
                close_price = open_price * random.uniform(0.98, 1.02)
                high_price = max(open_price, close_price) * random.uniform(1.0, 1.02)
                low_price = min(open_price, close_price) * random.uniform(0.98, 1.0)
                volume = random.randint(5000000, 15000000)
                
                data.append([
                    current_time - timedelta(days=limit-i),
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume
                ])
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            print(f"Error getting OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, order_type: str, side: str, 
                   quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a new order"""
        if not self.connected:
            return {'error': 'Not connected to Alpaca'}
        
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return {'error': f'Symbol {symbol} not found'}
            
            # Simulate order placement
            # In production: self.client.submit_order(symbol, quantity, side, order_type, 'day', limit_price=price)
            
            order_id = f"ALP_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            return {
                'id': order_id,
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': 'accepted',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.connected:
            return False
        
        try:
            # In production: self.client.cancel_order(order_id)
            print(f"Canceled Alpaca order: {order_id}")
            return True
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if not self.connected:
            return {}
        
        try:
            # In production: self.client.get_order(order_id)
            return {
                'id': order_id,
                'status': 'filled',
                'filled_qty': 100,
                'remaining_qty': 0
            }
        except Exception as e:
            print(f"Error getting order {order_id}: {e}")
            return {}

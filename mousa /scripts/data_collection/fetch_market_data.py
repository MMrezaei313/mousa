import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import sqlite3
import os

class MarketDataFetcher:
    def __init__(self, api_key=None, secret_key=None, db_path="data/market_data.db"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for storing market data"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                open_time DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                interval TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, open_time, interval)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT,
                published_at DATETIME,
                sentiment_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_binance_ohlcv(self, symbol='BTCUSDT', interval='1h', limit=500):
        """Fetch OHLCV data from Binance API"""
        try:
            base_url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df['symbol'] = symbol
            df['interval'] = interval
            
            self._save_to_database(df, symbol, interval)
            return df[['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'interval']]
            
        except Exception as e:
            print(f"Error fetching Binance data for {symbol}: {e}")
            return None
    
    def fetch_multiple_symbols(self, symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'], interval='1h', limit=100):
        """Fetch data for multiple symbols"""
        all_data = {}
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.fetch_binance_ohlcv(symbol, interval, limit)
            if data is not None:
                all_data[symbol] = data
            time.sleep(0.2)  # Rate limiting
        
        return all_data
    
    def _save_to_database(self, df, symbol, interval):
        """Save DataFrame to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            for _, row in df.iterrows():
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO ohlcv_data 
                    (symbol, open_time, open, high, low, close, volume, interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, row['open_time'], row['open'], row['high'], 
                    row['low'], row['close'], row['volume'], interval
                ))
            conn.commit()
            conn.close()
            print(f"Data for {symbol} saved to database.")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def fetch_historical_data(self, symbol, days=30, interval='1h'):
        """Fetch historical data for specified number of days"""
        limit = min(days * 24, 1000)  # Binance max limit
        return self.fetch_binance_ohlcv(symbol, interval, limit)
    
    def get_crypto_news(self, query='cryptocurrency', language='en', page_size=10):
        """Fetch cryptocurrency news from NewsAPI (requires API key)"""
        # This is a placeholder - you'll need to sign up for NewsAPI
        print("News API functionality requires NewsAPI key")
        return []

# Example usage
if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    btc_data = fetcher.fetch_binance_ohlcv('BTCUSDT', '1h', 100)
    if btc_data is not None:
        print(f"Fetched {len(btc_data)} records for BTCUSDT")
        print(btc_data.head())

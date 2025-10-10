import pandas as pd
import numpy as np
from fetch_market_data import MarketDataFetcher
import sqlite3
from datetime import datetime, timedelta
import os

class HistoricalDataManager:
    def __init__(self, db_path="data/market_data.db"):
        self.db_path = db_path
        self.fetcher = MarketDataFetcher(db_path=db_path)
    
    def update_historical_data(self, symbols=None, intervals=None, days=90):
        """Update historical data for specified symbols and intervals"""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT']
        
        if intervals is None:
            intervals = ['1h', '4h', '1d']
        
        for symbol in symbols:
            for interval in intervals:
                print(f"Updating {symbol} data for {interval} interval...")
                self.fetcher.fetch_binance_ohlcv(symbol, interval, 1000)
    
    def get_historical_data(self, symbol, start_date=None, end_date=None, interval='1h'):
        """Retrieve historical data from database"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT symbol, open_time, open, high, low, close, volume, interval
            FROM ohlcv_data 
            WHERE symbol = ? AND interval = ? AND open_time BETWEEN ? AND ?
            ORDER BY open_time ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=[symbol, interval, start_date, end_date])
        conn.close()
        
        if not df.empty:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df.set_index('open_time', inplace=True)
        
        return df
    
    def calculate_returns(self, df):
        """Calculate daily returns and volatility"""
        if df.empty:
            return df
        
        df['daily_return'] = df['close'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        df['volatility'] = df['daily_return'].rolling(window=20).std() * np.sqrt(365)
        
        return df
    
    def generate_market_report(self, symbols=None):
        """Generate a comprehensive market report"""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        report = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, interval='1d')
            if not df.empty:
                df = self.calculate_returns(df)
                
                latest = df.iloc[-1]
                report[symbol] = {
                    'current_price': latest['close'],
                    '24h_change': latest['daily_return'] * 100,
                    'volume': latest['volume'],
                    'volatility': latest['volatility'] if not pd.isna(latest['volatility']) else 0,
                    'support_level': df['low'].tail(20).min(),
                    'resistance_level': df['high'].tail(20).max()
                }
        
        return report

if __name__ == "__main__":
    manager = HistoricalDataManager()
    
    # Update data
    manager.update_historical_data()
    
    # Generate report
    report = manager.generate_market_report()
    for symbol, data in report.items():
        print(f"\n{symbol}:")
        for key, value in data.items():
            print(f"  {key}: {value:.4f}")

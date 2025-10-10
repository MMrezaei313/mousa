import websocket
import json
import threading
import time
import sqlite3
from datetime import datetime
import pandas as pd

class RealTimeDataStream:
    def __init__(self, db_path="data/market_data.db"):
        self.db_path = db_path
        self.ws = None
        self.is_connected = False
        self.symbols = ['btcusdt', 'ethusdt', 'adausdt']
        self.setup_database()
    
    def setup_database(self):
        """Setup database for real-time data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_time_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_price REAL NOT NULL,
                condition TEXT NOT NULL,
                is_triggered BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'stream' in data:
                stream_type = data['stream']
                stream_data = data['data']
                
                if 'k' in stream_data:  # Kline data
                    self.process_kline_data(stream_data)
                elif 's' in stream_data:  # Trade data
                    self.process_trade_data(stream_data)
                    
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def process_kline_data(self, data):
        """Process kline/candlestick data"""
        kline = data['k']
        symbol = kline['s']
        is_closed = kline['x']  # Is this kline closed?
        
        if is_closed:
            record = {
                'symbol': symbol,
                'open_time': datetime.fromtimestamp(kline['t'] / 1000),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'interval': kline['i']
            }
            
            self.save_kline_data(record)
            self.check_price_alerts(symbol, float(kline['c']))
    
    def process_trade_data(self, data):
        """Process real-time trade data"""
        symbol = data['s']
        price = float(data['p'])
        volume = float(data['q'])
        timestamp = datetime.fromtimestamp(data['T'] / 1000)
        
        self.save_trade_data(symbol, price, volume, timestamp)
        print(f"Trade: {symbol} - Price: {price}, Volume: {volume}")
    
    def save_kline_data(self, record):
        """Save kline data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ohlcv_data 
                (symbol, open_time, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['symbol'], record['open_time'], record['open'],
                record['high'], record['low'], record['close'],
                record['volume'], record['interval']
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving kline data: {e}")
    
    def save_trade_data(self, symbol, price, volume, timestamp):
        """Save trade data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO real_time_data (symbol, price, volume, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (symbol, price, volume, timestamp))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving trade data: {e}")
    
    def check_price_alerts(self, symbol, current_price):
        """Check if any price alerts have been triggered"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM price_alerts 
                WHERE symbol = ? AND is_triggered = FALSE
            ''', (symbol,))
            
            alerts = cursor.fetchall()
            
            for alert in alerts:
                alert_id, alert_symbol, alert_price, condition, is_triggered, created_at = alert
                
                triggered = False
                if condition == 'ABOVE' and current_price > alert_price:
                    triggered = True
                elif condition == 'BELOW' and current_price < alert_price:
                    triggered = True
                
                if triggered:
                    cursor.execute('''
                        UPDATE price_alerts SET is_triggered = TRUE WHERE id = ?
                    ''', (alert_id,))
                    print(f"🚨 PRICE ALERT: {symbol} {condition} {alert_price} - Current: {current_price}")
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error checking price alerts: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure"""
        print("WebSocket connection closed")
        self.is_connected = False
    
    def on_open(self, ws):
        """Handle WebSocket opening"""
        print("WebSocket connection opened")
        self.is_connected = True
        
        # Subscribe to streams
        streams = []
        for symbol in self.symbols:
            # Kline streams
            streams.append(f"{symbol}@kline_1m")
            streams.append(f"{symbol}@kline_5m")
            streams.append(f"{symbol}@kline_1h")
            # Trade streams
            streams.append(f"{symbol}@trade")
        
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        ws.send(json.dumps(subscribe_message))
    
    def start_stream(self):
        """Start the WebSocket stream"""
        def run():
            self.ws = websocket.WebSocketApp(
                "wss://stream.binance.com:9443/ws",
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            self.ws.run_forever()
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def add_price_alert(self, symbol, price, condition='ABOVE'):
        """Add a price alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO price_alerts (symbol, alert_price, condition)
                VALUES (?, ?, ?)
            ''', (symbol.upper(), price, condition.upper()))
            
            conn.commit()
            conn.close()
            print(f"Alert set: {symbol} {condition} {price}")
        except Exception as e:
            print(f"Error adding alert: {e}")
    
    def stop_stream(self):
        """Stop the WebSocket stream"""
        if self.ws:
            self.ws.close()

# Example usage
if __name__ == "__main__":
    stream = RealTimeDataStream()
    
    # Add some sample alerts
    stream.add_price_alert('BTCUSDT', 50000, 'ABOVE')
    stream.add_price_alert('BTCUSDT', 45000, 'BELOW')
    
    # Start streaming
    print("Starting real-time data stream...")
    stream.start_stream()
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping stream...")
        stream.stop_stream()

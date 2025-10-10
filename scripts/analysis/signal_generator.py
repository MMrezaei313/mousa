import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from technical_analysis import TechnicalAnalyzer
import json
import sqlite3

class TradingSignalGenerator:
    def __init__(self, db_path="data/market_data.db"):
        self.db_path = db_path
        self.analyzer = TechnicalAnalyzer()
        self.signals_history = []
    
    def generate_comprehensive_signals(self, symbol, timeframe='1h', lookback_periods=50):
        """Generate comprehensive trading signals for a symbol"""
        # Get historical data
        df = self._get_market_data(symbol, timeframe, lookback_periods)
        
        if df is None or df.empty:
            return None
        
        # Calculate technical indicators
        df = self.analyzer.calculate_all_indicators(df)
        
        # Generate signals
        signals_df = self.analyzer.generate_signals(df)
        
        if not signals_df.empty:
            latest_signal = signals_df.iloc[-1].to_dict()
            latest_signal['symbol'] = symbol
            latest_signal['timeframe'] = timeframe
            latest_signal['generated_at'] = datetime.now()
            
            # Save signal to database
            self._save_signal_to_db(latest_signal)
            
            return latest_signal
        
        return None
    
    def generate_multiple_timeframe_signals(self, symbol, timeframes=['15m', '1h', '4h']):
        """Generate signals across multiple timeframes"""
        signals = {}
        
        for tf in timeframes:
            signal = self.generate_comprehensive_signals(symbol, tf)
            if signal:
                signals[tf] = signal
        
        # Calculate consensus signal
        consensus = self._calculate_consensus_signal(signals)
        
        result = {
            'symbol': symbol,
            'timeframe_signals': signals,
            'consensus_signal': consensus,
            'analysis_timestamp': datetime.now()
        }
        
        return result
    
    def _calculate_consensus_signal(self, signals):
        """Calculate consensus signal from multiple timeframes"""
        if not signals:
            return 'NO_CONSENSUS'
        
        signal_weights = {
            '15m': 1,
            '1h': 2,
            '4h': 3,
            '1d': 4
        }
        
        signal_scores = {
            'STRONG_SELL': -2,
            'SELL': -1,
            'HOLD': 0,
            'BUY': 1,
            'STRONG_BUY': 2
        }
        
        total_weight = 0
        weighted_score = 0
        
        for tf, signal_data in signals.items():
            signal = signal_data.get('signal', 'HOLD')
            weight = signal_weights.get(tf, 1)
            
            if signal in signal_scores:
                weighted_score += signal_scores[signal] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 'NO_CONSENSUS'
        
        average_score = weighted_score / total_weight
        
        if average_score >= 1.5:
            return 'STRONG_BUY'
        elif average_score >= 0.5:
            return 'BUY'
        elif average_score <= -1.5:
            return 'STRONG_SELL'
        elif average_score <= -0.5:
            return 'SELL'
        else:
            return 'HOLD'
    
    def generate_portfolio_signals(self, symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT']):
        """Generate signals for entire portfolio"""
        portfolio_signals = {}
        
        for symbol in symbols:
            signal = self.generate_multiple_timeframe_signals(symbol)
            if signal:
                portfolio_signals[symbol] = signal
        
        # Calculate portfolio metrics
        portfolio_analysis = self._analyze_portfolio(portfolio_signals)
        
        return {
            'portfolio_signals': portfolio_signals,
            'portfolio_analysis': portfolio_analysis,
            'generated_at': datetime.now()
        }
    
    def _analyze_portfolio(self, portfolio_signals):
        """Analyze portfolio-level metrics"""
        if not portfolio_signals:
            return {}
        
        bullish_count = 0
        bearish_count = 0
        total_strength = 0
        symbols_count = len(portfolio_signals)
        
        for symbol, data in portfolio_signals.items():
            consensus = data.get('consensus_signal', 'HOLD')
            
            if consensus in ['BUY', 'STRONG_BUY']:
                bullish_count += 1
                total_strength += 1 if consensus == 'BUY' else 2
            elif consensus in ['SELL', 'STRONG_SELL']:
                bearish_count += 1
                total_strength -= 1 if consensus == 'SELL' else 2
        
        market_sentiment = 'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'NEUTRAL'
        
        return {
            'total_symbols': symbols_count,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'market_sentiment': market_sentiment,
            'sentiment_strength': total_strength / symbols_count if symbols_count > 0 else 0,
            'bullish_ratio': bullish_count / symbols_count if symbols_count > 0 else 0
        }
    
    def _get_market_data(self, symbol, timeframe, lookback_periods):
        """Retrieve market data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT open_time, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE symbol = ? AND interval = ?
                ORDER BY open_time DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=[symbol, timeframe, lookback_periods])
            conn.close()
            
            if not df.empty:
                df['open_time'] = pd.to_datetime(df['open_time'])
                df.set_index('open_time', inplace=True)
                df.sort_index(inplace=True)  # Ensure chronological order
            
            return df
            
        except Exception as e:
            print(f"Error retrieving market data: {e}")
            return None
    
    def _save_signal_to_db(self, signal):
        """Save generated signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    strength REAL,
                    reasons TEXT,
                    price REAL,
                    rsi REAL,
                    macd REAL,
                    timeframe TEXT,
                    generated_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                INSERT INTO trading_signals 
                (symbol, signal, strength, reasons, price, rsi, macd, timeframe, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'],
                signal['signal'],
                signal.get('strength', 0),
                signal.get('reasons', ''),
                signal.get('price', 0),
                signal.get('rsi', 0),
                signal.get('macd', 0),
                signal.get('timeframe', ''),
                signal.get('generated_at', datetime.now())
            ))
            
            conn.commit()
            conn.close()
            
            self.signals_history.append(signal)
            
        except Exception as e:
            print(f"Error saving signal to database: {e}")
    
    def get_signal_history(self, symbol=None, limit=50):
        """Retrieve signal history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if symbol:
                query = '''
                    SELECT * FROM trading_signals 
                    WHERE symbol = ?
                    ORDER BY generated_at DESC
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, params=[symbol, limit])
            else:
                query = '''
                    SELECT * FROM trading_signals 
                    ORDER BY generated_at DESC
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, params=[limit])
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error retrieving signal history: {e}")
            return pd.DataFrame()
    
    def calculate_signal_accuracy(self, symbol, days=30):
        """Calculate historical accuracy of signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT signal, price, generated_at
                FROM trading_signals 
                WHERE symbol = ? AND generated_at >= DATE('now', ?)
                ORDER BY generated_at
            '''
            
            df = pd.read_sql_query(query, conn, params=[symbol, f'-{days} days'])
            conn.close()
            
            if df.empty:
                return {'accuracy': 0, 'total_signals': 0, 'profitable_signals': 0}
            
            # This is a simplified accuracy calculation
            # In production, you'd compare signals with actual price movements
            profitable_count = 0
            
            for i in range(len(df) - 1):
                current_signal = df.iloc[i]
                next_price = df.iloc[i + 1]['price'] if i + 1 < len(df) else current_signal['price']
                
                # Simplified profitability check
                price_change = (next_price - current_signal['price']) / current_signal['price']
                
                if (current_signal['signal'] in ['BUY', 'STRONG_BUY'] and price_change > 0.01) or \
                   (current_signal['signal'] in ['SELL', 'STRONG_SELL'] and price_change < -0.01):
                    profitable_count += 1
            
            accuracy = profitable_count / (len(df) - 1) if len(df) > 1 else 0
            
            return {
                'accuracy': accuracy,
                'total_signals': len(df),
                'profitable_signals': profitable_count,
                'analysis_period_days': days
            }
            
        except Exception as e:
            print(f"Error calculating signal accuracy: {e}")
            return {'accuracy': 0, 'total_signals': 0, 'profitable_signals': 0}

# Example usage
if __name__ == "__main__":
    generator = TradingSignalGenerator()
    
    # Generate single symbol signal
    btc_signal = generator.generate_comprehensive_signals('BTCUSDT', '1h')
    print("BTC Signal:", btc_signal)
    
    # Generate multi-timeframe signals
    eth_multi = generator.generate_multiple_timeframe_signals('ETHUSDT')
    print("\nETH Multi-timeframe:", eth_multi['consensus_signal'])
    
    # Generate portfolio signals
    portfolio = generator.generate_portfolio_signals()
    print("\nPortfolio Analysis:", portfolio['portfolio_analysis'])
    
    # Get signal history
    history = generator.get_signal_history('BTCUSDT', 10)
    print(f"\nLast 10 BTC signals: {len(history)} signals found")

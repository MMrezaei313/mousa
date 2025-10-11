import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class StockAnalyzer:
    """
    Advanced stock market analysis tools
    """
    
    def __init__(self):
        self.sector_classification = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'NFLX'],
            'Financial Services': ['JPM', 'V', 'MA'],
            'Healthcare': ['JNJ', 'PFE', 'UNH'],
            'ETF': ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO']
        }
    
    def calculate_technical_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = ohlcv_data.copy()
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def analyze_sector_rotation(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze sector performance and rotation"""
        sector_performance = {}
        
        for sector, symbols in self.sector_classification.items():
            sector_returns = []
            for symbol in symbols:
                if symbol in stock_data and not stock_data[symbol].empty:
                    returns = stock_data[symbol]['close'].pct_change().mean() * 252  # Annualized
                    sector_returns.append(returns)
            
            if sector_returns:
                sector_performance[sector] = np.mean(sector_returns)
        
        return dict(sorted(sector_performance.items(), key=lambda x: x[1], reverse=True))
    
    def detect_market_sentiment(self, ohlcv_data: pd.DataFrame, news_sentiment: float = 0.0) -> Dict[str, Any]:
        """Detect overall market sentiment"""
        if ohlcv_data.empty:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        # Price-based sentiment
        price_trend = self._analyze_price_trend(ohlcv_data)
        
        # Volume-based sentiment
        volume_sentiment = self._analyze_volume_sentiment(ohlcv_data)
        
        # Technical indicator sentiment
        technical_sentiment = self._analyze_technical_sentiment(ohlcv_data)
        
        # Combine all factors
        sentiment_score = (price_trend + volume_sentiment + technical_sentiment + news_sentiment) / 4
        
        if sentiment_score > 0.3:
            sentiment = 'bullish'
        elif sentiment_score < -0.3:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'confidence': abs(sentiment_score),
            'factors': {
                'price_trend': price_trend,
                'volume_sentiment': volume_sentiment,
                'technical_sentiment': technical_sentiment,
                'news_sentiment': news_sentiment
            }
        }
    
    def _analyze_price_trend(self, ohlcv_data: pd.DataFrame) -> float:
        """Analyze price trend for sentiment"""
        prices = ohlcv_data['close']
        
        # Short-term vs long-term trend
        short_trend = prices.pct_change(5).mean()
        long_trend = prices.pct_change(20).mean()
        
        # Support/resistance breaks
        recent_high = prices.tail(10).max()
        recent_low = prices.tail(10).min()
        current_price = prices.iloc[-1]
        
        trend_score = (short_trend + long_trend) / 2
        
        # Bonus for breaking resistance
        if current_price >= recent_high:
            trend_score += 0.2
        # Penalty for breaking support
        elif current_price <= recent_low:
            trend_score -= 0.2
            
        return float(trend_score)
    
    def _analyze_volume_sentiment(self, ohlcv_data: pd.DataFrame) -> float:
        """Analyze volume for sentiment"""
        volume = ohlcv_data['volume']
        prices = ohlcv_data['close']
        
        # Volume trend
        volume_trend = volume.pct_change(5).mean()
        
        # Volume on up vs down days
        price_changes = prices.pct_change()
        up_volume = volume[price_changes > 0].mean()
        down_volume = volume[price_changes < 0].mean()
        
        if up_volume > down_volume:
            volume_score = 0.2
        else:
            volume_score = -0.2
            
        return float(volume_trend + volume_score)
    
    def _analyze_technical_sentiment(self, ohlcv_data: pd.DataFrame) -> float:
        """Analyze technical indicators for sentiment"""
        df = self.calculate_technical_indicators(ohlcv_data)
        
        if df.empty:
            return 0.0
        
        current = df.iloc[-1]
        
        technical_score = 0.0
        
        # RSI sentiment
        if current['rsi'] > 70:
            technical_score -= 0.3  # Overbought
        elif current['rsi'] < 30:
            technical_score += 0.3  # Oversold
        
        # MACD sentiment
        if current['macd'] > current['macd_signal']:
            technical_score += 0.2
        else:
            technical_score -= 0.2
            
        # Moving average sentiment
        if current['close'] > current['sma_20'] > current['sma_50']:
            technical_score += 0.3
        elif current['close'] < current['sma_20'] < current['sma_50']:
            technical_score -= 0.3
            
        return technical_score

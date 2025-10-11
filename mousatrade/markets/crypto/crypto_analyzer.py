import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class CryptoAnalyzer:
    """
    Advanced cryptocurrency market analysis
    """
    
    def __init__(self):
        self.technical_indicators = {}
    
    def calculate_volatility(self, ohlcv_data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volatility using standard deviation"""
        returns = ohlcv_data['close'].pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(365)  # Annualized
        return volatility
    
    def detect_market_regime(self, ohlcv_data: pd.DataFrame) -> str:
        """Detect current market regime (trending, ranging, volatile)"""
        volatility = self.calculate_volatility(ohlcv_data)
        current_volatility = volatility.iloc[-1] if not volatility.empty else 0
        
        # Simple trend detection
        prices = ohlcv_data['close']
        sma_20 = prices.rolling(window=20).mean()
        sma_50 = prices.rolling(window=50).mean()
        
        if current_volatility > 0.8:  # High volatility threshold
            return "high_volatility"
        elif sma_20.iloc[-1] > sma_50.iloc[-1] and prices.iloc[-1] > sma_20.iloc[-1]:
            return "bull_trend"
        elif sma_20.iloc[-1] < sma_50.iloc[-1] and prices.iloc[-1] < sma_20.iloc[-1]:
            return "bear_trend"
        else:
            return "ranging"
    
    def analyze_correlation(self, symbols: List[str], markets: Dict) -> pd.DataFrame:
        """Analyze correlation between different cryptocurrencies"""
        correlations = {}
        
        for symbol in symbols:
            try:
                ohlcv = markets['binance'].get_ohlcv(symbol, '1d', 100)
                if not ohlcv.empty:
                    correlations[symbol] = ohlcv['close'].pct_change().dropna()
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        corr_df = pd.DataFrame(correlations)
        return corr_df.corr()
    
    def find_arbitrage_opportunities(self, markets: List) -> List[Dict]:
        """Find arbitrage opportunities between different exchanges"""
        opportunities = []
        common_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        
        for symbol in common_symbols:
            prices = {}
            
            for market in markets:
                try:
                    ticker = market.get_ticker(symbol)
                    if ticker and 'bid' in ticker and 'ask' in ticker:
                        prices[market.name] = {
                            'bid': ticker['bid'],
                            'ask': ticker['ask']
                        }
                except Exception as e:
                    continue
            
            # Find price differences
            if len(prices) >= 2:
                exchanges = list(prices.keys())
                for i in range(len(exchanges)):
                    for j in range(i + 1, len(exchanges)):
                        exch1, exch2 = exchanges[i], exchanges[j]
                        price_diff = abs(prices[exch1]['bid'] - prices[exch2]['ask'])
                        spread_percent = (price_diff / prices[exch1]['bid']) * 100
                        
                        if spread_percent > 0.5:  # 0.5% threshold
                            opportunities.append({
                                'symbol': symbol,
                                'buy_exchange': exch2 if prices[exch2]['ask'] < prices[exch1]['bid'] else exch1,
                                'sell_exchange': exch1 if prices[exch2]['ask'] < prices[exch1]['bid'] else exch2,
                                'spread_percent': spread_percent,
                                'potential_profit': price_diff
                            })
        
        return sorted(opportunities, key=lambda x: x['spread_percent'], reverse=True)
    
    def calculate_support_resistance(self, ohlcv_data: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate support and resistance levels"""
        highs = ohlcv_data['high'].rolling(window=window).max()
        lows = ohlcv_data['low'].rolling(window=window).min()
        
        current_high = highs.iloc[-1] if not highs.empty else 0
        current_low = lows.iloc[-1] if not lows.empty else 0
        current_price = ohlcv_data['close'].iloc[-1] if not ohlcv_data.empty else 0
        
        return {
            'support': current_low,
            'resistance': current_high,
            'distance_to_support': ((current_price - current_low) / current_low * 100) if current_low > 0 else 0,
            'distance_to_resistance': ((current_high - current_price) / current_price * 100) if current_price > 0 else 0
        }

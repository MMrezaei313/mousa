import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class VolatilityCalculator:
    """
    Advanced volatility calculation for multiple markets
    """
    
    def __init__(self):
        self.volatility_cache = {}
    
    def calculate_historical_volatility(self, prices: pd.Series, period: int = 20, 
                                      annualize: bool = True) -> float:
        """Calculate historical volatility"""
        returns = prices.pct_change().dropna()
        if len(returns) < period:
            return 0.0
        
        volatility = returns.rolling(window=period).std().iloc[-1]
        
        if annualize:
            # Assuming 252 trading days for stocks, 365 for crypto/forex
            trading_days = 252 if len(prices) <= 365 else 365
            volatility *= np.sqrt(trading_days)
        
        return volatility
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_rolling_volatility(self, ohlcv_data: pd.DataFrame, 
                                   window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        returns = ohlcv_data['close'].pct_change()
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return rolling_vol
    
    def detect_volatility_regime(self, ohlcv_data: pd.DataFrame, 
                               lookback_period: int = 50) -> str:
        """Detect current volatility regime"""
        volatility = self.calculate_rolling_volatility(ohlcv_data)
        current_vol = volatility.iloc[-1] if not volatility.empty else 0
        historical_vol = volatility.mean()
        
        if current_vol > historical_vol * 1.5:
            return "high_volatility"
        elif current_vol < historical_vol * 0.7:
            return "low_volatility"
        else:
            return "normal_volatility"
    
    def calculate_volatility_breakout(self, ohlcv_data: pd.DataFrame, 
                                    period: int = 20) -> Dict[str, float]:
        """Calculate volatility breakout levels"""
        atr = self.calculate_atr(ohlcv_data['high'], ohlcv_data['low'], 
                               ohlcv_data['close'], period)
        current_atr = atr.iloc[-1] if not atr.empty else 0
        current_close = ohlcv_data['close'].iloc[-1] if not ohlcv_data.empty else 0
        
        return {
            'upper_breakout': current_close + (current_atr * 2),
            'lower_breakout': current_close - (current_atr * 2),
            'atr': current_atr,
            'atr_percentage': (current_atr / current_close) * 100 if current_close > 0 else 0
        }
    
    def compare_volatility_across_markets(self, market_data: Dict[str, pd.DataFrame],
                                        symbols: List[str]) -> pd.DataFrame:
        """Compare volatility across different markets and symbols"""
        volatility_data = {}
        
        for symbol, data in market_data.items():
            if symbol in symbols and not data.empty:
                vol = self.calculate_historical_volatility(data['close'])
                volatility_data[symbol] = vol
        
        return pd.DataFrame.from_dict(volatility_data, orient='index', 
                                    columns=['volatility']).sort_values('volatility', ascending=False)

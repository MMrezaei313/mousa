import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy
    """
    
    def __init__(self, fast_window: int = 10, slow_window: int = 30, **kwargs):
        params = {
            'fast_window': fast_window,
            'slow_window': slow_window
        }
        params.update(kwargs)
        super().__init__("MovingAverageCrossover", **params)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MA crossover"""
        fast_ma = data['close'].rolling(window=self.params['fast_window']).mean()
        slow_ma = data['close'].rolling(window=self.params['slow_window']).mean()
        
        # Generate signals (1 for long, -1 for short, 0 for neutral)
        signals = pd.DataFrame(index=data.index)
        signals['fast_ma'] = fast_ma
        signals['slow_ma'] = slow_ma
        signals['position'] = np.where(fast_ma > slow_ma, 1, -1)
        signals['crossover'] = (
            (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        ) | (
            (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        )
        
        self.signals = signals
        return signals

class RSIMeanReversion(BaseStrategy):
    """
    RSI Mean Reversion Strategy
    """
    
    def __init__(self, rsi_period: int = 14, oversold: int = 30, 
                 overbought: int = 70, **kwargs):
        params = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought
        }
        params.update(kwargs)
        super().__init__("RSIMeanReversion", **params)
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI levels"""
        rsi = self.calculate_rsi(data['close'])
        
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = rsi
        
        # Generate signals
        long_signal = rsi < self.params['oversold']
        short_signal = rsi > self.params['overbought']
        
        signals['position'] = 0
        signals.loc[long_signal, 'position'] = 1
        signals.loc[short_signal, 'position'] = -1
        
        # Add signal strength based on RSI extremity
        signals['signal_strength'] = np.where(
            rsi < self.params['oversold'],
            (self.params['oversold'] - rsi) / self.params['oversold'],
            np.where(
                rsi > self.params['overbought'],
                (rsi - self.params['overbought']) / (100 - self.params['overbought']),
                0
            )
        )
        
        self.signals = signals
        return signals

class MACDStrategy(BaseStrategy):
    """
    MACD Strategy
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, **kwargs):
        params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }
        params.update(kwargs)
        super().__init__("MACDStrategy", **params)
        
    def calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=self.params['fast_period'], adjust=False).mean()
        exp2 = prices.ewm(span=self.params['slow_period'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.params['signal_period'], adjust=False).mean()
        histogram = macd - signal
        
        return pd.DataFrame({
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }, index=prices.index)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MACD"""
        macd_data = self.calculate_macd(data['close'])
        
        signals = pd.DataFrame(index=data.index)
        signals['macd'] = macd_data['macd']
        signals['signal_line'] = macd_data['signal']
        signals['histogram'] = macd_data['histogram']
        
        # Generate signals based on MACD crossover
        signals['position'] = np.where(
            macd_data['macd'] > macd_data['signal'], 1, -1
        )
        
        # Identify crossover points
        signals['crossover'] = (
            (macd_data['macd'] > macd_data['signal']) & 
            (macd_data['macd'].shift(1) <= macd_data['signal'].shift(1))
        ) | (
            (macd_data['macd'] < macd_data['signal']) & 
            (macd_data['macd'].shift(1) >= macd_data['signal'].shift(1))
        )
        
        self.signals = signals
        return signals

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0, **kwargs):
        params = {
            'period': period,
            'num_std': num_std
        }
        params.update(kwargs)
        super().__init__("BollingerBandsStrategy", **params)
        
    def calculate_bollinger_bands(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        middle_band = prices.rolling(window=self.params['period']).mean()
        std = prices.rolling(window=self.params['period']).std()
        
        upper_band = middle_band + (std * self.params['num_std'])
        lower_band = middle_band - (std * self.params['num_std'])
        
        return pd.DataFrame({
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band
        }, index=prices.index)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands"""
        bb_data = self.calculate_bollinger_bands(data['close'])
        
        signals = pd.DataFrame(index=data.index)
        signals['middle_band'] = bb_data['middle_band']
        signals['upper_band'] = bb_data['upper_band']
        signals['lower_band'] = bb_data['lower_band']
        signals['price'] = data['close']
        
        # Calculate position within bands (0 to 1)
        signals['band_position'] = (
            (data['close'] - signals['lower_band']) / 
            (signals['upper_band'] - signals['lower_band'])
        )
        
        # Generate mean reversion signals
        signals['position'] = np.where(
            data['close'] < signals['lower_band'], 1,  # Oversold - buy
            np.where(
                data['close'] > signals['upper_band'], -1,  # Overbought - sell
                0  # Neutral
            )
        )
        
        self.signals = signals
        return signals

# Strategy factory function
def create_strategy(strategy_name: str, **params) -> BaseStrategy:
    """Factory function to create strategy instances"""
    strategies = {
        'moving_average_crossover': MovingAverageCrossover,
        'rsi_mean_reversion': RSIMeanReversion,
        'macd': MACDStrategy,
        'bollinger_bands': BollingerBandsStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategies[strategy_name](**params)

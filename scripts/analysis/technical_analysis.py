import pandas as pd
import numpy as np
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df):
        """Calculate all technical indicators"""
        if df.empty:
            return df
        
        # Trend Indicators
        df = self._add_trend_indicators(df)
        
        # Momentum Indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility Indicators
        df = self._add_volatility_indicators(df)
        
        # Volume Indicators
        df = self._add_volume_indicators(df)
        
        # Support and Resistance
        df = self._calculate_support_resistance(df)
        
        return df
    
    def _add_trend_indicators(self, df):
        """Add trend-based indicators"""
        # Moving Averages
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
        
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Ichimoku Cloud (simplified)
        df = self._calculate_ichimoku(df)
        
        return df
    
    def _add_momentum_indicators(self, df):
        """Add momentum-based indicators"""
        # RSI
        df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
        df['rsi_7'] = RSIIndicator(close=df['close'], window=7).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # CCI
        df['cci'] = self._calculate_cci(df)
        
        return df
    
    def _add_volatility_indicators(self, df):
        """Add volatility-based indicators"""
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR
        df['atr_14'] = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()
        
        return df
    
    def _add_volume_indicators(self, df):
        """Add volume-based indicators"""
        # OBV
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        # VWAP
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
        ).volume_weighted_average_price()
        
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def _calculate_ichimoku(self, df, conversion_period=9, base_period=26, lagging_span=52, displacement=26):
        """Calculate Ichimoku Cloud indicators"""
        # Conversion Line (Tenkan-sen)
        high_9 = df['high'].rolling(window=conversion_period).max()
        low_9 = df['low'].rolling(window=conversion_period).min()
        df['ichimoku_conversion'] = (high_9 + low_9) / 2
        
        # Base Line (Kijun-sen)
        high_26 = df['high'].rolling(window=base_period).max()
        low_26 = df['low'].rolling(window=base_period).min()
        df['ichimoku_base'] = (high_26 + low_26) / 2
        
        # Leading Span A (Senkou Span A)
        df['ichimoku_span_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(displacement)
        
        # Leading Span B (Senkou Span B)
        high_52 = df['high'].rolling(window=lagging_span).max()
        low_52 = df['low'].rolling(window=lagging_span).min()
        df['ichimoku_span_b'] = ((high_52 + low_52) / 2).shift(displacement)
        
        # Lagging Span (Chikou Span)
        df['ichimoku_lagging'] = df['close'].shift(-displacement)
        
        return df
    
    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        return -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    
    def _calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (typical_price - sma) / (0.015 * mad)
    
    def _calculate_support_resistance(self, df, window=20):
        """Calculate dynamic support and resistance levels"""
        df['support'] = df['low'].rolling(window=window).min()
        df['resistance'] = df['high'].rolling(window=window).max()
        
        # Pivot Points
        df = self._calculate_pivot_points(df)
        
        return df
    
    def _calculate_pivot_points(self, df):
        """Calculate classic pivot points"""
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        return df
    
    def generate_signals(self, df):
        """Generate comprehensive trading signals"""
        if df.empty:
            return pd.DataFrame()
        
        signals = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            signal_strength = 0
            signal_type = 'HOLD'
            reasons = []
            
            # Trend Analysis
            if current['sma_20'] > current['sma_50'] and prev['sma_20'] <= prev['sma_50']:
                signal_strength += 2
                reasons.append("Golden Cross (SMA 20 > SMA 50)")
            
            if current['sma_20'] < current['sma_50'] and prev['sma_20'] >= prev['sma_50']:
                signal_strength -= 2
                reasons.append("Death Cross (SMA 20 < SMA 50)")
            
            # MACD Signals
            if (prev['macd'] <= prev['macd_signal'] and 
                current['macd'] > current['macd_signal']):
                signal_strength += 1
                reasons.append("MACD Bullish Crossover")
            
            if (prev['macd'] >= prev['macd_signal'] and 
                current['macd'] < current['macd_signal']):
                signal_strength -= 1
                reasons.append("MACD Bearish Crossover")
            
            # RSI Signals
            if current['rsi_14'] < 30:
                signal_strength += 1
                reasons.append("RSI Oversold")
            elif current['rsi_14'] > 70:
                signal_strength -= 1
                reasons.append("RSI Overbought")
            
            # Bollinger Bands
            if current['close'] < current['bb_lower']:
                signal_strength += 1
                reasons.append("Below Lower Bollinger Band")
            elif current['close'] > current['bb_upper']:
                signal_strength -= 1
                reasons.append("Above Upper Bollinger Band")
            
            # Volume Confirmation
            if current['volume'] > current['volume_sma_20'] * 1.5:
                if signal_strength > 0:
                    signal_strength += 0.5
                    reasons.append("High Volume Confirmation")
                elif signal_strength < 0:
                    signal_strength -= 0.5
                    reasons.append("High Volume Confirmation")
            
            # Determine final signal
            if signal_strength >= 3:
                signal_type = 'STRONG_BUY'
            elif signal_strength >= 1.5:
                signal_type = 'BUY'
            elif signal_strength <= -3:
                signal_type = 'STRONG_SELL'
            elif signal_strength <= -1.5:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            
            signals.append({
                'timestamp': current.name if hasattr(current, 'name') else i,
                'signal': signal_type,
                'strength': abs(signal_strength),
                'reasons': '; '.join(reasons),
                'price': current['close'],
                'rsi': current['rsi_14'],
                'macd': current['macd']
            })
        
        return pd.DataFrame(signals)

    def calculate_risk_metrics(self, df):
        """Calculate risk management metrics"""
        if df.empty:
            return {}
        
        returns = df['close'].pct_change().dropna()
        
        metrics = {
            'volatility': returns.std() * np.sqrt(365),  # Annualized
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0,
            'max_drawdown': (df['close'] / df['close'].cummax() - 1).min(),
            'var_95': returns.quantile(0.05),
            'current_rsi': df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else None,
            'bb_position': (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / 
                          (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) if 'bb_upper' in df.columns else None
        }
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.normal(50000, 1000, 100),
        'high': np.random.normal(50500, 1000, 100),
        'low': np.random.normal(49500, 1000, 100),
        'close': np.random.normal(50000, 1000, 100),
        'volume': np.random.normal(1000, 200, 100)
    }, index=dates)
    
    analyzer = TechnicalAnalyzer()
    analyzed_data = analyzer.calculate_all_indicators(sample_data)
    signals = analyzer.generate_signals(analyzed_data)
    risk_metrics = analyzer.calculate_risk_metrics(analyzed_data)
    
    print("Technical Analysis Completed!")
    print(f"Generated {len(signals)} signals")
    print("Recent signals:")
    print(signals.tail())
    print("\nRisk Metrics:")
    for key, value in risk_metrics.items():
        print(f"{key}: {value:.4f}")

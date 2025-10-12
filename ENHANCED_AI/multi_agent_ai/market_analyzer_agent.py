# market_analyzer_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ta  # Technical analysis library

class MarketCondition(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"

@dataclass
class MarketAnalysisResult:
    condition: MarketCondition
    confidence: float
    key_indicators: Dict[str, float]
    anomalies: List[Dict]
    support_levels: List[float]
    resistance_levels: List[float]
    trend_strength: float
    volatility: float

class MarketAnalyzerAgent:
    """
    AI Agent for comprehensive market condition analysis
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Technical indicator parameters
        self.indicators = {
            'rsi_period': 14,
            'bb_period': 20,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14
        }
    
    def analyze_market_conditions(self, 
                                price_data: pd.DataFrame,
                                volume_data: pd.DataFrame = None,
                                fundamental_data: Dict = None) -> MarketAnalysisResult:
        """
        Comprehensive market condition analysis
        
        Args:
            price_data: DataFrame with OHLC data
            volume_data: DataFrame with volume data
            fundamental_data: Fundamental analysis data
            
        Returns:
            MarketAnalysisResult object
        """
        try:
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(price_data)
            
            # Detect market condition
            condition, confidence = self._detect_market_condition(indicators, price_data)
            
            # Find support and resistance levels
            support_levels, resistance_levels = self._find_support_resistance(price_data)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(price_data, indicators)
            
            # Calculate trend strength and volatility
            trend_strength = self._calculate_trend_strength(price_data)
            volatility = self._calculate_volatility(price_data)
            
            return MarketAnalysisResult(
                condition=condition,
                confidence=confidence,
                key_indicators=indicators,
                anomalies=anomalies,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_strength=trend_strength,
                volatility=volatility
            )
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            raise
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        # RSI
        indicators['rsi'] = ta.momentum.RSIIndicator(
            data['close'], window=self.indicators['rsi_period']
        ).rsi().iloc[-1]
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            data['close'], window=self.indicators['bb_period']
        )
        indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
        indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
        indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        
        # MACD
        macd = ta.trend.MACD(
            data['close'], 
            window_slow=self.indicators['macd_slow'],
            window_fast=self.indicators['macd_fast'],
            window_sign=self.indicators['macd_signal']
        )
        indicators['macd'] = macd.macd().iloc[-1]
        indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
        
        # ATR (Average True Range)
        atr = ta.volatility.AverageTrueRange(
            data['high'], data['low'], data['close'], 
            window=self.indicators['atr_period']
        )
        indicators['atr'] = atr.average_true_range().iloc[-1]
        
        # Moving Averages
        indicators['sma_20'] = ta.trend.SMAIndicator(data['close'], window=20).sma_indicator().iloc[-1]
        indicators['sma_50'] = ta.trend.SMAIndicator(data['close'], window=50).sma_indicator().iloc[-1]
        indicators['ema_12'] = ta.trend.EMAIndicator(data['close'], window=12).ema_indicator().iloc[-1]
        indicators['ema_26'] = ta.trend.EMAIndicator(data['close'], window=26).ema_indicator().iloc[-1]
        
        # Price position relative to MAs
        current_price = data['close'].iloc[-1]
        indicators['price_vs_sma20'] = (current_price - indicators['sma_20']) / indicators['sma_20'] * 100
        indicators['price_vs_sma50'] = (current_price - indicators['sma_50']) / indicators['sma_50'] * 100
        
        return indicators
    
    def _detect_market_condition(self, 
                               indicators: Dict[str, float], 
                               price_data: pd.DataFrame) -> Tuple[MarketCondition, float]:
        """Detect current market condition with confidence score"""
        
        conditions = []
        confidences = []
        current_price = price_data['close'].iloc[-1]
        
        # RSI-based condition
        rsi = indicators['rsi']
        if rsi > 70:
            conditions.append(MarketCondition.BEARISH)
            confidences.append(min((rsi - 70) / 30, 1.0))
        elif rsi < 30:
            conditions.append(MarketCondition.BULLISH)
            confidences.append(min((30 - rsi) / 30, 1.0))
        
        # Moving Average-based condition
        price_vs_sma20 = indicators['price_vs_sma20']
        if abs(price_vs_sma20) < 2:  # Within 2% of SMA20
            conditions.append(MarketCondition.SIDEWAYS)
            confidences.append(0.7)
        elif price_vs_sma20 > 5:  # More than 5% above SMA20
            conditions.append(MarketCondition.BULLISH)
            confidences.append(0.8)
        elif price_vs_sma20 < -5:  # More than 5% below SMA20
            conditions.append(MarketCondition.BEARISH)
            confidences.append(0.8)
        
        # Volatility-based condition
        bb_width = indicators['bb_width']
        if bb_width > 0.05:  # High volatility
            conditions.append(MarketCondition.VOLATILE)
            confidences.append(min(bb_width * 10, 1.0))
        
        # Trend strength
        if indicators['macd_histogram'] > 0 and indicators['ema_12'] > indicators['ema_26']:
            conditions.append(MarketCondition.TRENDING)
            confidences.append(0.6)
        
        # Determine final condition (most frequent with highest average confidence)
        if conditions:
            condition_counts = {}
            condition_confidences = {}
            
            for cond, conf in zip(conditions, confidences):
                if cond not in condition_counts:
                    condition_counts[cond] = 0
                    condition_confidences[cond] = []
                condition_counts[cond] += 1
                condition_confidences[cond].append(conf)
            
            # Find condition with highest frequency and confidence
            best_condition = max(condition_counts.items(), key=lambda x: x[1])[0]
            avg_confidence = np.mean(condition_confidences[best_condition])
            
            return best_condition, min(avg_confidence, 1.0)
        else:
            return MarketCondition.SIDEWAYS, 0.5
    
    def _find_support_resistance(self, 
                               data: pd.DataFrame, 
                               window: int = 20) -> Tuple[List[float], List[float]]:
        """Find key support and resistance levels using swing points"""
        
        highs = data['high']
        lows = data['low']
        
        # Find local maxima (resistance)
        resistance_levels = []
        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i-window:i+window].max():
                resistance_levels.append(highs.iloc[i])
        
        # Find local minima (support)
        support_levels = []
        for i in range(window, len(lows) - window):
            if lows.iloc[i] == lows.iloc[i-window:i+window].min():
                support_levels.append(lows.iloc[i])
        
        # Remove duplicates and sort
        support_levels = sorted(list(set(support_levels)))
        resistance_levels = sorted(list(set(resistance_levels)))
        
        return support_levels[-3:], resistance_levels[-3:]  # Return last 3 levels
    
    def _detect_anomalies(self, 
                         price_data: pd.DataFrame, 
                         indicators: Dict[str, float]) -> List[Dict]:
        """Detect anomalous market behavior"""
        
        anomalies = []
        current_price = price_data['close'].iloc[-1]
        
        # Price spike detection
        price_returns = price_data['close'].pct_change().dropna()
        recent_return = price_returns.iloc[-1]
        
        if abs(recent_return) > price_returns.std() * 3:
            anomalies.append({
                'type': 'price_spike',
                'severity': 'high',
                'description': f'Significant price movement: {recent_return:.2%}',
                'value': recent_return
            })
        
        # Volume spike detection (if volume data available)
        if 'volume' in price_data.columns:
            volume = price_data['volume']
            volume_anomaly = volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 2
            if volume_anomaly:
                anomalies.append({
                    'type': 'volume_spike',
                    'severity': 'medium',
                    'description': 'Unusual trading volume detected',
                    'value': volume.iloc[-1]
                })
        
        # RSI extreme
        if indicators['rsi'] > 80 or indicators['rsi'] < 20:
            anomalies.append({
                'type': 'rsi_extreme',
                'severity': 'medium',
                'description': f'RSI at extreme level: {indicators["rsi"]:.1f}',
                'value': indicators['rsi']
            })
        
        return anomalies
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX"""
        try:
            adx = ta.trend.ADXIndicator(
                data['high'], data['low'], data['close'], window=14
            )
            adx_value = adx.adx().iloc[-1]
            # Normalize to 0-1 range
            return min(adx_value / 100, 1.0)
        except:
            return 0.5
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate normalized volatility"""
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        # Normalize (assuming 50% is very high volatility)
        return min(volatility / 0.5, 1.0)
    
    def get_market_summary(self, analysis_result: MarketAnalysisResult) -> Dict:
        """Generate human-readable market summary"""
        
        summary = {
            'market_condition': analysis_result.condition.value,
            'confidence_score': analysis_result.confidence,
            'key_findings': [],
            'recommendations': []
        }
        
        # Add key findings
        if analysis_result.anomalies:
            summary['key_findings'].append(
                f"Detected {len(analysis_result.anomalies)} market anomalies"
            )
        
        summary['key_findings'].append(
            f"Trend strength: {analysis_result.trend_strength:.1%}"
        )
        summary['key_findings'].append(
            f"Volatility: {analysis_result.volatility:.1%}"
        )
        
        # Add recommendations based on condition
        if analysis_result.condition == MarketCondition.BULLISH:
            summary['recommendations'].append("Consider long positions with proper risk management")
        elif analysis_result.condition == MarketCondition.BEARISH:
            summary['recommendations'].append("Consider short positions or reduce exposure")
        elif analysis_result.condition == MarketCondition.VOLATILE:
            summary['recommendations'].append("Use smaller position sizes and wider stops")
        
        return summary

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    analyzer = MarketAnalyzerAgent()
    result = analyzer.analyze_market_conditions(sample_data)
    summary = analyzer.get_market_summary(result)
    
    print("Market Analysis Result:")
    print(f"Condition: {result.condition.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Trend Strength: {result.trend_strength:.2f}")
    print(f"Volatility: {result.volatility:.2f}")
    print(f"Anomalies detected: {len(result.anomalies)}")
    print(f"Support Levels: {result.support_levels}")
    print(f"Resistance Levels: {result.resistance_levels}")

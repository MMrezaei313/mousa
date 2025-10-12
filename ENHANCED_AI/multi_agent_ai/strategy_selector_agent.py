# strategy_selector_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend" 
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERSION = "mean_reversion"

class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    HEDGING = "hedging"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"

@dataclass
class StrategyPerformance:
    strategy_type: StrategyType
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float

@dataclass
class StrategyRecommendation:
    strategy_type: StrategyType
    confidence: float
    expected_return: float
    expected_risk: float
    position_size: float
    time_horizon: str
    market_regime: MarketRegime
    rationale: str
    risk_level: str

@dataclass
class StrategySelection:
    primary_strategy: StrategyRecommendation
    secondary_strategies: List[StrategyRecommendation]
    avoidance_strategies: List[StrategyType]
    regime_analysis: Dict[str, float]
    market_conditions: Dict[str, Any]

class StrategySelectorAgent:
    """
    Advanced AI Agent for dynamic strategy selection based on market conditions
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'min_confidence_threshold': 0.65,
            'max_drawdown_limit': 0.15,
            'risk_free_rate': 0.02,
            'lookback_period': 252,  # 1 year
            'regime_lookback': 63,   # 3 months
            'strategy_weights': {
                'performance': 0.35,
                'regime_fit': 0.25,
                'risk_adjusted': 0.20,
                'market_conditions': 0.20
            }
        }
        
        self.scaler = StandardScaler()
        self.regime_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.strategy_performance_history = {}
        self.market_regime_history = []
        
        # Strategy-regime mapping
        self.regime_strategy_map = {
            MarketRegime.BULL_TREND: [
                StrategyType.TREND_FOLLOWING, 
                StrategyType.MOMENTUM,
                StrategyType.BREAKOUT
            ],
            MarketRegime.BEAR_TREND: [
                StrategyType.TREND_FOLLOWING,
                StrategyType.HEDGING,
                StrategyType.BREAKOUT
            ],
            MarketRegime.SIDEWAYS: [
                StrategyType.MEAN_REVERSION,
                StrategyType.MARKET_MAKING,
                StrategyType.SCALPING
            ],
            MarketRegime.HIGH_VOLATILITY: [
                StrategyType.BREAKOUT,
                StrategyType.SWING_TRADING,
                StrategyType.HEDGING
            ],
            MarketRegime.LOW_VOLATILITY: [
                StrategyType.MEAN_REVERSION,
                StrategyType.SCALPING,
                StrategyType.MARKET_MAKING
            ],
            MarketRegime.TRENDING: [
                StrategyType.TREND_FOLLOWING,
                StrategyType.MOMENTUM,
                StrategyType.SWING_TRADING
            ],
            MarketRegime.MEAN_REVERSION: [
                StrategyType.MEAN_REVERSION,
                StrategyType.SCALPING,
                StrategyType.MARKET_MAKING
            ]
        }
    
    def select_optimal_strategy(self,
                              price_data: pd.DataFrame,
                              volume_data: pd.DataFrame = None,
                              market_indicators: Dict = None,
                              historical_performance: Dict[StrategyType, StrategyPerformance] = None,
                              risk_tolerance: str = "moderate") -> StrategySelection:
        """
        Select optimal trading strategy based on current market conditions
        
        Args:
            price_data: OHLC price data
            volume_data: Volume data
            market_indicators: Additional market indicators
            historical_performance: Historical performance of strategies
            risk_tolerance: User risk tolerance (low, moderate, high)
            
        Returns:
            StrategySelection object
        """
        try:
            # Analyze current market regime
            current_regime, regime_confidence = self._analyze_market_regime(price_data, volume_data)
            
            # Calculate market condition metrics
            market_conditions = self._analyze_market_conditions(price_data, volume_data, market_indicators)
            
            # Evaluate strategy performance
            strategy_scores = self._evaluate_strategies(
                current_regime, market_conditions, historical_performance, risk_tolerance
            )
            
            # Select primary and secondary strategies
            primary_strategy, secondary_strategies, avoidance_strategies = self._select_strategies(
                strategy_scores, current_regime, risk_tolerance
            )
            
            # Generate regime analysis
            regime_analysis = self._generate_regime_analysis(current_regime, regime_confidence, market_conditions)
            
            return StrategySelection(
                primary_strategy=primary_strategy,
                secondary_strategies=secondary_strategies,
                avoidance_strategies=avoidance_strategies,
                regime_analysis=regime_analysis,
                market_conditions=market_conditions
            )
            
        except Exception as e:
            self.logger.error(f"Error in strategy selection: {e}")
            raise
    
    def _analyze_market_regime(self, 
                             price_data: pd.DataFrame,
                             volume_data: pd.DataFrame = None) -> Tuple[MarketRegime, float]:
        """Analyze current market regime using multiple indicators"""
        
        returns = price_data['close'].pct_change().dropna()
        
        # Calculate regime indicators
        trend_strength = self._calculate_trend_strength(price_data)
        volatility = self._calculate_volatility(returns)
        mean_reversion_tendency = self._calculate_mean_reversion_tendency(price_data)
        momentum = self._calculate_momentum(price_data)
        
        # Regime classification logic
        if trend_strength > 0.7 and momentum > 0:
            regime = MarketRegime.BULL_TREND
            confidence = min(trend_strength, 0.9)
        elif trend_strength > 0.7 and momentum < 0:
            regime = MarketRegime.BEAR_TREND
            confidence = min(trend_strength, 0.9)
        elif volatility > 0.8:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(volatility, 0.85)
        elif volatility < 0.3:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(1 - volatility, 0.85)
        elif mean_reversion_tendency > 0.6:
            regime = MarketRegime.MEAN_REVERSION
            confidence = min(mean_reversion_tendency, 0.8)
        elif trend_strength > 0.5:
            regime = MarketRegime.TRENDING
            confidence = min(trend_strength, 0.75)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.6
        
        # Store regime history for ML training
        self.market_regime_history.append({
            'regime': regime,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'mean_reversion': mean_reversion_tendency,
            'momentum': momentum
        })
        
        return regime, confidence
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX and moving averages"""
        try:
            from ta.trend import ADXIndicator
            high, low, close = price_data['high'], price_data['low'], price_data['close']
            adx = ADXIndicator(high, low, close, window=14)
            adx_value = adx.adx().iloc[-1]
            return min(adx_value / 100, 1.0)  # Normalize to 0-1
        except:
            # Fallback calculation
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            trend_alignment = (sma_20.iloc[-1] > sma_50.iloc[-1]) and (sma_20.iloc[-5] > sma_50.iloc[-5])
            return 0.7 if trend_alignment else 0.3
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate normalized volatility"""
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return min(volatility / 0.5, 1.0)  # Normalize (50% = very high)
    
    def _calculate_mean_reversion_tendency(self, price_data: pd.DataFrame) -> float:
        """Calculate tendency for mean reversion using RSI and Bollinger Bands"""
        try:
            from ta.momentum import RSIIndicator
            from ta.volatility import BollingerBands
            
            rsi = RSIIndicator(price_data['close'], window=14).rsi().iloc[-1]
            bb = BollingerBands(price_data['close'], window=20)
            bb_position = (price_data['close'].iloc[-1] - bb.bollinger_mavg().iloc[-1]) / (
                bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])
            
            # Mean reversion score
            rsi_score = 1 - abs(rsi - 50) / 50  # Closer to 50 = higher mean reversion
            bb_score = 1 - abs(bb_position)      # Closer to middle = higher mean reversion
            
            return (rsi_score + bb_score) / 2
        except:
            return 0.5
    
    def _calculate_momentum(self, price_data: pd.DataFrame) -> float:
        """Calculate price momentum"""
        returns_1m = price_data['close'].pct_change(21).iloc[-1]  # 1-month
        returns_3m = price_data['close'].pct_change(63).iloc[-1]  # 3-month
        
        # Combined momentum score
        momentum_score = (returns_1m + returns_3m) / 2
        return np.tanh(momentum_score * 10)  # Normalize to -1 to 1
    
    def _analyze_market_conditions(self,
                                 price_data: pd.DataFrame,
                                 volume_data: pd.DataFrame = None,
                                 market_indicators: Dict = None) -> Dict[str, Any]:
        """Comprehensive market condition analysis"""
        
        returns = price_data['close'].pct_change().dropna()
        current_price = price_data['close'].iloc[-1]
        
        conditions = {
            'volatility': self._calculate_volatility(returns),
            'trend_strength': self._calculate_trend_strength(price_data),
            'momentum': self._calculate_momentum(price_data),
            'mean_reversion_tendency': self._calculate_mean_reversion_tendency(price_data),
            'liquidity': self._calculate_liquidity(price_data, volume_data),
            'market_sentiment': self._calculate_market_sentiment(price_data, market_indicators),
            'support_resistance_strength': self._calculate_support_resistance_strength(price_data),
            'volume_profile': self._analyze_volume_profile(price_data, volume_data)
        }
        
        return conditions
    
    def _calculate_liquidity(self, 
                           price_data: pd.DataFrame, 
                           volume_data: pd.DataFrame = None) -> float:
        """Calculate market liquidity score"""
        if volume_data is not None and 'volume' in volume_data.columns:
            volume = volume_data['volume']
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            return min(current_volume / avg_volume, 2.0) / 2.0  # Normalize to 0-1
        return 0.5
    
    def _calculate_market_sentiment(self,
                                  price_data: pd.DataFrame,
                                  market_indicators: Dict = None) -> float:
        """Calculate market sentiment score"""
        # Simplified sentiment calculation
        # In practice, this would use news sentiment, fear-greed index, etc.
        price_position = (price_data['close'].iloc[-1] - price_data['low'].min()) / (
            price_data['high'].max() - price_data['low'].min())
        return price_position
    
    def _calculate_support_resistance_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate strength of support/resistance levels"""
        # Simplified implementation
        high, low, close = price_data['high'], price_data['low'], price_data['close']
        resistance_hits = (high == high.rolling(20).max()).sum()
        support_hits = (low == low.rolling(20).min()).sum()
        total_touches = resistance_hits + support_hits
        return min(total_touches / len(price_data) * 10, 1.0)
    
    def _analyze_volume_profile(self,
                              price_data: pd.DataFrame,
                              volume_data: pd.DataFrame = None) -> Dict[str, float]:
        """Analyze volume profile characteristics"""
        if volume_data is None or 'volume' not in volume_data.columns:
            return {'volume_trend': 0.5, 'volume_volatility': 0.5}
        
        volume = volume_data['volume']
        volume_trend = volume.rolling(10).mean().pct_change(5).iloc[-1]
        volume_volatility = volume.pct_change().std()
        
        return {
            'volume_trend': np.tanh(volume_trend * 10),  # Normalize
            'volume_volatility': min(volume_volatility * 10, 1.0)
        }
    
    def _evaluate_strategies(self,
                           current_regime: MarketRegime,
                           market_conditions: Dict[str, Any],
                           historical_performance: Dict[StrategyType, StrategyPerformance],
                           risk_tolerance: str) -> Dict[StrategyType, float]:
        """Evaluate and score all available strategies"""
        
        strategy_scores = {}
        
        for strategy in StrategyType:
            # Base score from regime alignment
            regime_score = self._calculate_regime_alignment(strategy, current_regime)
            
            # Market condition fit
            condition_score = self._calculate_condition_fit(strategy, market_conditions)
            
            # Historical performance
            performance_score = self._calculate_performance_score(strategy, historical_performance)
            
            # Risk adjustment
            risk_score = self._calculate_risk_adjustment(strategy, risk_tolerance, historical_performance)
            
            # Combined weighted score
            weights = self.config['strategy_weights']
            total_score = (
                regime_score * weights['regime_fit'] +
                condition_score * weights['market_conditions'] +
                performance_score * weights['performance'] +
                risk_score * weights['risk_adjusted']
            )
            
            strategy_scores[strategy] = total_score
        
        return strategy_scores
    
    def _calculate_regime_alignment(self, strategy: StrategyType, regime: MarketRegime) -> float:
        """Calculate how well strategy aligns with current regime"""
        recommended_strategies = self.regime_strategy_map.get(regime, [])
        
        if strategy in recommended_strategies:
            return 0.9  # Highly aligned
        elif any(s in recommended_strategies for s in self._get_related_strategies(strategy)):
            return 0.7  # Moderately aligned
        else:
            return 0.3  # Poorly aligned
    
    def _get_related_strategies(self, strategy: StrategyType) -> List[StrategyType]:
        """Get strategies related to the given strategy"""
        related_map = {
            StrategyType.TREND_FOLLOWING: [StrategyType.MOMENTUM, StrategyType.SWING_TRADING],
            StrategyType.MEAN_REVERSION: [StrategyType.SCALPING, StrategyType.MARKET_MAKING],
            StrategyType.MOMENTUM: [StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT],
            StrategyType.BREAKOUT: [StrategyType.MOMENTUM, StrategyType.TREND_FOLLOWING],
            StrategyType.SCALPING: [StrategyType.MARKET_MAKING, StrategyType.MEAN_REVERSION],
        }
        return related_map.get(strategy, [])
    
    def _calculate_condition_fit(self, strategy: StrategyType, conditions: Dict[str, Any]) -> float:
        """Calculate how well strategy fits current market conditions"""
        
        fit_scores = []
        
        # Trend-based strategies
        if strategy in [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM, StrategyType.BREAKOUT]:
            fit_scores.append(conditions['trend_strength'])
            fit_scores.append(conditions['momentum'])
        
        # Mean reversion strategies
        if strategy in [StrategyType.MEAN_REVERSION, StrategyType.SCALPING, StrategyType.MARKET_MAKING]:
            fit_scores.append(conditions['mean_reversion_tendency'])
            fit_scores.append(1 - conditions['trend_strength'])  # Inverse relationship
        
        # Volatility strategies
        if strategy in [StrategyType.BREAKOUT, StrategyType.SWING_TRADING]:
            fit_scores.append(conditions['volatility'])
        
        # Liquidity-sensitive strategies
        if strategy in [StrategyType.SCALPING, StrategyType.MARKET_MAKING]:
            fit_scores.append(conditions['liquidity'])
        
        return np.mean(fit_scores) if fit_scores else 0.5
    
    def _calculate_performance_score(self,
                                  strategy: StrategyType,
                                  historical_performance: Dict[StrategyType, StrategyPerformance]) -> float:
        """Calculate performance-based score"""
        if not historical_performance or strategy not in historical_performance:
            return 0.5  # Neutral score for unknown performance
        
        perf = historical_performance[strategy]
        
        # Multi-factor performance scoring
        sharpe_score = min(perf.sharpe_ratio / 2.0, 1.0)  # Normalize (2.0 = excellent)
        drawdown_score = 1 - min(perf.max_drawdown / 0.3, 1.0)  # Inverse (30% = terrible)
        win_rate_score = perf.win_rate
        profit_factor_score = min(perf.profit_factor / 3.0, 1.0)  # Normalize (3.0 = excellent)
        
        performance_score = (sharpe_score + drawdown_score + win_rate_score + profit_factor_score) / 4
        return performance_score
    
    def _calculate_risk_adjustment(self,
                                strategy: StrategyType,
                                risk_tolerance: str,
                                historical_performance: Dict[StrategyType, StrategyPerformance]) -> float:
        """Adjust score based on risk tolerance compatibility"""
        
        strategy_risk_levels = {
            StrategyType.SCALPING: "high",
            StrategyType.MOMENTUM: "high", 
            StrategyType.BREAKOUT: "high",
            StrategyType.TREND_FOLLOWING: "moderate",
            StrategyType.SWING_TRADING: "moderate",
            StrategyType.MEAN_REVERSION: "moderate",
            StrategyType.MARKET_MAKING: "low",
            StrategyType.HEDGING: "low",
            StrategyType.ARBITRAGE: "low"
        }
        
        strategy_risk = strategy_risk_levels.get(strategy, "moderate")
        
        # Risk compatibility matrix
        compatibility = {
            ("low", "low"): 0.9,
            ("low", "moderate"): 0.7,
            ("low", "high"): 0.3,
            ("moderate", "low"): 0.7,
            ("moderate", "moderate"): 0.9,
            ("moderate", "high"): 0.7,
            ("high", "low"): 0.3,
            ("high", "moderate"): 0.7,
            ("high", "high"): 0.9
        }
        
        return compatibility.get((risk_tolerance, strategy_risk), 0.5)
    
    def _select_strategies(self,
                         strategy_scores: Dict[StrategyType, float],
                         current_regime: MarketRegime,
                         risk_tolerance: str) -> Tuple[StrategyRecommendation, List[StrategyRecommendation], List[StrategyType]]:
        """Select primary, secondary, and avoidance strategies"""
        
        # Sort strategies by score
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select primary strategy (highest score above threshold)
        primary_strategy_type, primary_score = sorted_strategies[0]
        if primary_score < self.config['min_confidence_threshold']:
            self.logger.warning("No strategy meets confidence threshold")
        
        # Create primary strategy recommendation
        primary_recommendation = StrategyRecommendation(
            strategy_type=primary_strategy_type,
            confidence=primary_score,
            expected_return=self._estimate_expected_return(primary_strategy_type, current_regime),
            expected_risk=self._estimate_expected_risk(primary_strategy_type, risk_tolerance),
            position_size=self._calculate_position_size(primary_strategy_type, risk_tolerance),
            time_horizon=self._get_strategy_time_horizon(primary_strategy_type),
            market_regime=current_regime,
            rationale=self._generate_strategy_rationale(primary_strategy_type, current_regime),
            risk_level=self._get_strategy_risk_level(primary_strategy_type)
        )
        
        # Select secondary strategies (next 2-3 best)
        secondary_recommendations = []
        for strategy_type, score in sorted_strategies[1:4]:
            if score > self.config['min_confidence_threshold'] - 0.1:  # Slightly lower threshold
                recommendation = StrategyRecommendation(
                    strategy_type=strategy_type,
                    confidence=score,
                    expected_return=self._estimate_expected_return(strategy_type, current_regime),
                    expected_risk=self._estimate_expected_risk(strategy_type, risk_tolerance),
                    position_size=self._calculate_position_size(strategy_type, risk_tolerance),
                    time_horizon=self._get_strategy_time_horizon(strategy_type),
                    market_regime=current_regime,
                    rationale=self._generate_strategy_rationale(strategy_type, current_regime),
                    risk_level=self._get_strategy_risk_level(strategy_type)
                )
                secondary_recommendations.append(recommendation)
        
        # Identify strategies to avoid (lowest scores)
        avoidance_strategies = [strategy for strategy, score in sorted_strategies[-3:] 
                              if score < self.config['min_confidence_threshold'] - 0.2]
        
        return primary_recommendation, secondary_recommendations, avoidance_strategies
    
    def _estimate_expected_return(self, strategy: StrategyType, regime: MarketRegime) -> float:
        """Estimate expected return for strategy in current regime"""
        # Base expected returns by strategy-regime combination
        return_estimates = {
            (StrategyType.TREND_FOLLOWING, MarketRegime.BULL_TREND): 0.15,
            (StrategyType.TREND_FOLLOWING, MarketRegime.BEAR_TREND): 0.12,
            (StrategyType.MEAN_REVERSION, MarketRegime.SIDEWAYS): 0.08,
            (StrategyType.MOMENTUM, MarketRegime.TRENDING): 0.18,
            (StrategyType.BREAKOUT, MarketRegime.HIGH_VOLATILITY): 0.20,
            (StrategyType.SCALPING, MarketRegime.LOW_VOLATILITY): 0.06,
        }
        
        return return_estimates.get((strategy, regime), 0.10)  # Default 10%
    
    def _estimate_expected_risk(self, strategy: StrategyType, risk_tolerance: str) -> float:
        """Estimate expected risk for strategy"""
        base_risk = {
            StrategyType.SCALPING: 0.08,
            StrategyType.MOMENTUM: 0.15,
            StrategyType.BREAKOUT: 0.18,
            StrategyType.TREND_FOLLOWING: 0.12,
            StrategyType.SWING_TRADING: 0.10,
            StrategyType.MEAN_REVERSION: 0.09,
            StrategyType.MARKET_MAKING: 0.06,
            StrategyType.HEDGING: 0.05,
            StrategyType.ARBITRAGE: 0.04
        }
        
        risk_multiplier = {
            "low": 0.8,
            "moderate": 1.0,
            "high": 1.2
        }
        
        return base_risk.get(strategy, 0.10) * risk_multiplier.get(risk_tolerance, 1.0)
    
    def _calculate_position_size(self, strategy: StrategyType, risk_tolerance: str) -> float:
        """Calculate recommended position size"""
        base_sizes = {
            "low": 0.02,    # 2% of portfolio
            "moderate": 0.05, # 5% of portfolio  
            "high": 0.08     # 8% of portfolio
        }
        
        strategy_multipliers = {
            StrategyType.SCALPING: 0.5,
            StrategyType.MOMENTUM: 1.2,
            StrategyType.BREAKOUT: 1.5,
            StrategyType.TREND_FOLLOWING: 1.0,
            StrategyType.SWING_TRADING: 0.8,
            StrategyType.MEAN_REVERSION: 0.7
        }
        
        base_size = base_sizes.get(risk_tolerance, 0.05)
        multiplier = strategy_multipliers.get(strategy, 1.0)
        
        return base_size * multiplier
    
    def _get_strategy_time_horizon(self, strategy: StrategyType) -> str:
        """Get typical time horizon for strategy"""
        time_horizons = {
            StrategyType.SCALPING: "seconds_minutes",
            StrategyType.MARKET_MAKING: "minutes_hours", 
            StrategyType.ARBITRAGE: "minutes_hours",
            StrategyType.MEAN_REVERSION: "hours_days",
            StrategyType.MOMENTUM: "days_weeks",
            StrategyType.BREAKOUT: "days_weeks",
            StrategyType.TREND_FOLLOWING: "weeks_months",
            StrategyType.SWING_TRADING: "weeks_months",
            StrategyType.HEDGING: "months"
        }
        return time_horizons.get(strategy, "days_weeks")
    
    def _generate_strategy_rationale(self, strategy: StrategyType, regime: MarketRegime) -> str:
        """Generate rationale for strategy recommendation"""
        rationales = {
            (StrategyType.TREND_FOLLOWING, MarketRegime.BULL_TREND): 
                "Strong bullish trend detected - trend following strategies typically perform well in sustained uptrends",
            (StrategyType.MEAN_REVERSION, MarketRegime.SIDEWAYS):
                "Market showing sideways movement with mean reversion characteristics - range-bound strategies recommended",
            (StrategyType.BREAKOUT, MarketRegime.HIGH_VOLATILITY):
                "High volatility environment favorable for breakout strategies as price movements become more pronounced"
        }
        
        return rationales.get((strategy, regime), 
                            f"{strategy.value.replace('_', ' ').title()} strategy selected based on current market analysis")
    
    def _get_strategy_risk_level(self, strategy: StrategyType) -> str:
        """Get risk level for strategy"""
        risk_levels = {
            StrategyType.SCALPING: "high",
            StrategyType.MOMENTUM: "high",
            StrategyType.BREAKOUT: "high", 
            StrategyType.TREND_FOLLOWING: "moderate",
            StrategyType.SWING_TRADING: "moderate",
            StrategyType.MEAN_REVERSION: "moderate",
            StrategyType.MARKET_MAKING: "low",
            StrategyType.HEDGING: "low",
            StrategyType.ARBITRAGE: "low"
        }
        return risk_levels.get(strategy, "moderate")
    
    def _generate_regime_analysis(self, 
                                regime: MarketRegime, 
                                confidence: float,
                                market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Generate detailed regime analysis"""
        return {
            'current_regime': regime.value,
            'regime_confidence': confidence,
            'trend_strength': market_conditions['trend_strength'],
            'volatility_level': market_conditions['volatility'],
            'momentum_score': market_conditions['momentum'],
            'mean_reversion_score': market_conditions['mean_reversion_tendency'],
            'liquidity_score': market_conditions['liquidity']
        }

# Example usage
if __name__ == "__main__":
    # Sample price data
    price_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    })
    
    # Sample historical performance
    historical_performance = {
        StrategyType.TREND_FOLLOWING: StrategyPerformance(
            strategy_type=StrategyType.TREND_FOLLOWING,
            sharpe_ratio=1.5,
            max_drawdown=0.12,
            win_rate=0.55,
            profit_factor=1.8,
            total_return=0.25,
            volatility=0.18,
            calmar_ratio=2.08,
            sortino_ratio=2.1
        ),
        StrategyType.MEAN_REVERSION: StrategyPerformance(
            strategy_type=StrategyType.MEAN_REVERSION,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            win_rate=0.60,
            profit_factor=1.5,
            total_return=0.15,
            volatility=0.12,
            calmar_ratio=1.88,
            sortino_ratio=1.6
        )
    }
    
    selector = StrategySelectorAgent()
    selection = selector.select_optimal_strategy(
        price_data=price_data,
        historical_performance=historical_performance,
        risk_tolerance="moderate"
    )
    
    print("Strategy Selection Results:")
    print(f"Primary Strategy: {selection.primary_strategy.strategy_type.value}")
    print(f"Confidence: {selection.primary_strategy.confidence:.2f}")
    print(f"Expected Return: {selection.primary_strategy.expected_return:.2%}")
    print(f"Risk Level: {selection.primary_strategy.risk_level}")
    print(f"Rationale: {selection.primary_strategy.rationale}")
    print(f"Market Regime: {selection.regime_analysis['current_regime']}")
    print(f"Secondary Strategies: {[s.strategy_type.value for s in selection.secondary_strategies]}")
    print(f"Avoidance Strategies: {[s.value for s in selection.avoidance_strategies]}")

# risk_assessor_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class PositionType(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    beta: float
    correlation_matrix: pd.DataFrame
    stress_test_results: Dict[str, float]

@dataclass
class PositionRisk:
    position_id: str
    symbol: str
    position_type: PositionType
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    leverage: float = 1.0

@dataclass
class RiskAssessment:
    overall_risk: RiskLevel
    portfolio_var: float
    max_position_risk: float
    concentration_risk: float
    liquidity_risk: float
    market_risk: float
    credit_risk: float
    operational_risk: float
    risk_metrics: RiskMetrics
    recommendations: List[str]
    warnings: List[str]
    required_actions: List[str]

class RiskAssessorAgent:
    """
    Advanced AI Agent for comprehensive risk assessment and management
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'max_portfolio_var': 0.02,  # 2% max VaR
            'max_drawdown_limit': 0.15,  # 15% max drawdown
            'max_position_size': 0.10,   # 10% max per position
            'max_leverage': 5.0,
            'risk_free_rate': 0.02,
            'confidence_levels': [0.95, 0.99],
            'stress_scenarios': {
                'market_crash': -0.20,
                'volatility_spike': 0.50,
                'liquidity_crisis': -0.15
            }
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: 0.02,
            RiskLevel.LOW: 0.05,
            RiskLevel.MODERATE: 0.10,
            RiskLevel.HIGH: 0.15,
            RiskLevel.VERY_HIGH: 0.25,
            RiskLevel.EXTREME: 0.35
        }
    
    def assess_portfolio_risk(self,
                            portfolio: List[PositionRisk],
                            market_data: Dict[str, pd.DataFrame],
            historical_returns: pd.DataFrame,
            correlation_data: pd.DataFrame = None) -> RiskAssessment:
        """
        Comprehensive portfolio risk assessment
        
        Args:
            portfolio: List of position risks
            market_data: Current market data for all positions
            historical_returns: Historical returns data
            correlation_data: Asset correlation matrix
            
        Returns:
            RiskAssessment object
        """
        try:
            # Calculate basic risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio, historical_returns, correlation_data)
            
            # Assess various risk types
            market_risk = self._assess_market_risk(portfolio, market_data, risk_metrics)
            concentration_risk = self._assess_concentration_risk(portfolio)
            liquidity_risk = self._assess_liquidity_risk(portfolio, market_data)
            credit_risk = self._assess_credit_risk(portfolio)
            operational_risk = self._assess_operational_risk(portfolio)
            
            # Calculate overall risk level
            overall_risk = self._calculate_overall_risk(
                market_risk, concentration_risk, liquidity_risk, 
                credit_risk, operational_risk, risk_metrics
            )
            
            # Generate recommendations and warnings
            recommendations, warnings, required_actions = self._generate_risk_guidance(
                portfolio, overall_risk, risk_metrics
            )
            
            return RiskAssessment(
                overall_risk=overall_risk,
                portfolio_var=risk_metrics.var_95,
                max_position_risk=self._calculate_max_position_risk(portfolio),
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                market_risk=market_risk,
                credit_risk=credit_risk,
                operational_risk=operational_risk,
                risk_metrics=risk_metrics,
                recommendations=recommendations,
                warnings=warnings,
                required_actions=required_actions
            )
            
        except Exception as e:
            self.logger.error(f"Error in portfolio risk assessment: {e}")
            raise
    
    def _calculate_risk_metrics(self, 
                              portfolio: List[PositionRisk],
                              historical_returns: pd.DataFrame,
                              correlation_data: pd.DataFrame = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_returns)
        
        # VaR Calculation (Historical Simulation)
        var_95 = self._calculate_var(portfolio_returns, 0.95)
        var_99 = self._calculate_var(portfolio_returns, 0.99)
        
        # Expected Shortfall (CVaR)
        expected_shortfall = self._calculate_expected_shortfall(portfolio_returns, 0.95)
        
        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        # Risk-Adjusted Returns
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        
        # Volatility
        volatility = portfolio_returns.std()
        
        # Beta (if benchmark provided)
        beta = self._calculate_beta(portfolio_returns, historical_returns)
        
        # Correlation Matrix
        correlation_matrix = self._calculate_correlation_matrix(portfolio, historical_returns, correlation_data)
        
        # Stress Testing
        stress_test_results = self._perform_stress_test(portfolio, historical_returns)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            volatility=volatility,
            beta=beta,
            correlation_matrix=correlation_matrix,
            stress_test_results=stress_test_results
        )
    
    def _calculate_portfolio_returns(self, 
                                   portfolio: List[PositionRisk],
                                   historical_returns: pd.DataFrame) -> pd.Series:
        """Calculate weighted portfolio returns"""
        
        # Calculate position weights
        total_value = sum(pos.quantity * pos.current_price for pos in portfolio)
        weights = []
        assets = []
        
        for position in portfolio:
            weight = (position.quantity * position.current_price) / total_value
            weights.append(weight)
            assets.append(position.symbol)
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=historical_returns.index)
        
        for i, asset in enumerate(assets):
            if asset in historical_returns.columns:
                portfolio_returns += weights[i] * historical_returns[asset]
        
        return portfolio_returns
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk using historical simulation"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        return tail_returns.mean() if len(tail_returns) > 0 else var
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.config['risk_free_rate'] / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - self.config['risk_free_rate'] / 252
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
    
    def _calculate_beta(self, portfolio_returns: pd.Series, market_returns: pd.DataFrame) -> float:
        """Calculate portfolio beta relative to market"""
        if 'market' in market_returns.columns:
            market_data = market_returns['market']
            covariance = portfolio_returns.cov(market_data)
            market_variance = market_data.var()
            return covariance / market_variance if market_variance > 0 else 1.0
        return 1.0
    
    def _calculate_correlation_matrix(self, 
                                   portfolio: List[PositionRisk],
                                   historical_returns: pd.DataFrame,
                                   correlation_data: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate asset correlation matrix"""
        assets = [pos.symbol for pos in portfolio if pos.symbol in historical_returns.columns]
        
        if correlation_data is not None:
            return correlation_data.loc[assets, assets]
        else:
            return historical_returns[assets].corr()
    
    def _perform_stress_test(self, 
                           portfolio: List[PositionRisk],
                           historical_returns: pd.DataFrame) -> Dict[str, float]:
        """Perform stress testing under various scenarios"""
        
        stress_results = {}
        
        for scenario, impact in self.config['stress_scenarios'].items():
            # Simulate portfolio impact under stress scenario
            stressed_returns = historical_returns * (1 + impact)
            stressed_portfolio_returns = self._calculate_portfolio_returns(portfolio, stressed_returns)
            stress_var = self._calculate_var(stressed_portfolio_returns, 0.95)
            stress_results[scenario] = stress_var
        
        return stress_results
    
    def _assess_market_risk(self, 
                          portfolio: List[PositionRisk],
                          market_data: Dict[str, pd.DataFrame],
                          risk_metrics: RiskMetrics) -> float:
        """Assess market risk exposure"""
        
        # Calculate weighted market risk
        total_risk = 0.0
        total_value = sum(pos.quantity * pos.current_price for pos in portfolio)
        
        for position in portfolio:
            if position.symbol in market_data:
                asset_data = market_data[position.symbol]
                position_value = position.quantity * position.current_price
                weight = position_value / total_value
                
                # Calculate position-specific risk metrics
                asset_volatility = asset_data['close'].pct_change().std()
                leverage_factor = min(position.leverage, self.config['max_leverage'])
                
                position_risk = asset_volatility * leverage_factor * weight
                total_risk += position_risk
        
        return min(total_risk, 1.0)
    
    def _assess_concentration_risk(self, portfolio: List[PositionRisk]) -> float:
        """Assess portfolio concentration risk"""
        
        if not portfolio:
            return 0.0
        
        total_value = sum(pos.quantity * pos.current_price for pos in portfolio)
        position_values = [(pos.quantity * pos.current_price) for pos in portfolio]
        
        # Herfindahl-Hirschman Index (HHI) for concentration
        hhi = sum((value / total_value) ** 2 for value in position_values)
        
        # Normalize to 0-1 scale
        max_hhi = 1.0  # Complete concentration
        min_hhi = 1 / len(portfolio)  # Perfect diversification
        
        concentration_risk = (hhi - min_hhi) / (max_hhi - min_hhi)
        return min(concentration_risk, 1.0)
    
    def _assess_liquidity_risk(self, 
                             portfolio: List[PositionRisk],
                             market_data: Dict[str, pd.DataFrame]) -> float:
        """Assess liquidity risk"""
        
        liquidity_scores = []
        
        for position in portfolio:
            if position.symbol in market_data:
                asset_data = market_data[position.symbol]
                
                # Calculate liquidity metrics
                if 'volume' in asset_data.columns:
                    avg_volume = asset_data['volume'].mean()
                    position_volume = position.quantity
                    
                    # Days to liquidate (assuming 10% of average daily volume)
                    days_to_liquidate = position_volume / (avg_volume * 0.1) if avg_volume > 0 else 10
                    liquidity_score = min(days_to_liquidate / 10, 1.0)  # Normalize
                    liquidity_scores.append(liquidity_score)
        
        return np.mean(liquidity_scores) if liquidity_scores else 0.5
    
    def _assess_credit_risk(self, portfolio: List[PositionRisk]) -> float:
        """Assess credit/counterparty risk"""
        # Simplified credit risk assessment
        # In practice, this would involve credit ratings, CDS spreads, etc.
        return 0.1  # Base level assumption
    
    def _assess_operational_risk(self, portfolio: List[PositionRisk]) -> float:
        """Assess operational risk"""
        # Simplified operational risk assessment
        complexity_penalty = min(len(portfolio) * 0.01, 0.2)
        leverage_penalty = sum(min(pos.leverage * 0.05, 0.1) for pos in portfolio) / len(portfolio) if portfolio else 0
        return 0.05 + complexity_penalty + leverage_penalty
    
    def _calculate_overall_risk(self, 
                              market_risk: float,
                              concentration_risk: float,
                              liquidity_risk: float,
                              credit_risk: float,
                              operational_risk: float,
                              risk_metrics: RiskMetrics) -> RiskLevel:
        """Calculate overall risk level"""
        
        # Weighted risk score
        weights = {
            'market_risk': 0.4,
            'concentration_risk': 0.2,
            'liquidity_risk': 0.15,
            'credit_risk': 0.15,
            'operational_risk': 0.1
        }
        
        weighted_risk = (
            market_risk * weights['market_risk'] +
            concentration_risk * weights['concentration_risk'] +
            liquidity_risk * weights['liquidity_risk'] +
            credit_risk * weights['credit_risk'] +
            operational_risk * weights['operational_risk']
        )
        
        # Adjust for VaR and drawdown
        var_adjustment = min(risk_metrics.var_95 / self.config['max_portfolio_var'], 2.0)
        drawdown_adjustment = min(abs(risk_metrics.max_drawdown) / self.config['max_drawdown_limit'], 2.0)
        
        final_risk_score = weighted_risk * (var_adjustment + drawdown_adjustment) / 2
        
        # Map to risk levels
        for risk_level, threshold in self.risk_thresholds.items():
            if final_risk_score <= threshold:
                return risk_level
        
        return RiskLevel.EXTREME
    
    def _calculate_max_position_risk(self, portfolio: List[PositionRisk]) -> float:
        """Calculate maximum individual position risk"""
        if not portfolio:
            return 0.0
        
        total_value = sum(pos.quantity * pos.current_price for pos in portfolio)
        max_position_value = max(pos.quantity * pos.current_price for pos in portfolio)
        return max_position_value / total_value
    
    def _generate_risk_guidance(self,
                              portfolio: List[PositionRisk],
                              overall_risk: RiskLevel,
                              risk_metrics: RiskMetrics) -> Tuple[List[str], List[str], List[str]]:
        """Generate risk management recommendations and warnings"""
        
        recommendations = []
        warnings = []
        required_actions = []
        
        # Risk level based guidance
        if overall_risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
            warnings.append(f"Portfolio risk level is {overall_risk.value.upper()}. Consider reducing exposure.")
            required_actions.append("Review and potentially reduce position sizes")
        
        # VaR based guidance
        if risk_metrics.var_95 > self.config['max_portfolio_var']:
            warnings.append(f"Portfolio VaR ({risk_metrics.var_95:.2%}) exceeds limit ({self.config['max_portfolio_var']:.2%})")
            recommendations.append("Diversify portfolio to reduce VaR")
        
        # Drawdown based guidance
        if abs(risk_metrics.max_drawdown) > self.config['max_drawdown_limit']:
            warnings.append(f"Maximum drawdown ({abs(risk_metrics.max_drawdown):.2%}) exceeds limit ({self.config['max_drawdown_limit']:.2%})")
            recommendations.append("Implement stricter stop-loss levels")
        
        # Concentration risk guidance
        max_position_risk = self._calculate_max_position_risk(portfolio)
        if max_position_risk > self.config['max_position_size']:
            warnings.append(f"Maximum position size ({max_position_risk:.2%}) exceeds limit ({self.config['max_position_size']:.2%})")
            required_actions.append("Reduce largest position sizes")
        
        # Leverage guidance
        high_leverage_positions = [pos for pos in portfolio if pos.leverage > self.config['max_leverage']]
        if high_leverage_positions:
            warnings.append(f"{len(high_leverage_positions)} positions exceed maximum leverage")
            required_actions.append("Reduce leverage on high-risk positions")
        
        # Positive recommendations
        if risk_metrics.sharpe_ratio > 1.0:
            recommendations.append("Good risk-adjusted returns detected")
        
        if len(portfolio) >= 5:
            recommendations.append("Portfolio shows good diversification")
        
        return recommendations, warnings, required_actions
    
    def calculate_position_size(self,
                              symbol: str,
                              account_balance: float,
                              risk_per_trade: float = 0.02,
                              stop_loss_pct: float = 0.05,
                              volatility_adjustment: bool = True) -> Dict[str, float]:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            account_balance: Total account balance
            risk_per_trade: Risk per trade as percentage of account
            stop_loss_pct: Stop loss percentage
            volatility_adjustment: Whether to adjust for volatility
            
        Returns:
            Dictionary with position sizing details
        """
        
        max_risk_amount = account_balance * risk_per_trade
        position_value = max_risk_amount / stop_loss_pct
        
        if volatility_adjustment:
            # Adjust position size based on volatility (simplified)
            volatility_factor = 1.0  # Would be calculated from historical volatility
            position_value *= volatility_factor
        
        return {
            'max_position_value': position_value,
            'risk_amount': max_risk_amount,
            'position_size_percent': position_value / account_balance,
            'recommended_stop_loss': stop_loss_pct
        }

# Example usage
if __name__ == "__main__":
    # Sample portfolio
    portfolio = [
        PositionRisk(
            position_id="1",
            symbol="AAPL",
            position_type=PositionType.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            stop_loss=140.0,
            take_profit=170.0,
            leverage=1.0
        ),
        PositionRisk(
            position_id="2", 
            symbol="GOOGL",
            position_type=PositionType.LONG,
            quantity=50,
            entry_price=2500.0,
            current_price=2550.0,
            stop_loss=2400.0,
            take_profit=2700.0,
            leverage=1.0
        )
    ]
    
    # Sample market data
    market_data = {
        "AAPL": pd.DataFrame({
            'close': [150, 152, 155, 153, 155],
            'volume': [1000000, 1200000, 1100000, 1050000, 1150000]
        }),
        "GOOGL": pd.DataFrame({
            'close': [2500, 2520, 2550, 2530, 2550],
            'volume': [500000, 520000, 510000, 505000, 515000]
        })
    }
    
    # Sample historical returns
    historical_returns = pd.DataFrame({
        'AAPL': [0.01, 0.02, -0.01, 0.03, 0.01],
        'GOOGL': [0.02, 0.01, 0.03, -0.02, 0.02],
        'market': [0.015, 0.012, 0.01, 0.005, 0.015]
    })
    
    risk_agent = RiskAssessorAgent()
    assessment = risk_agent.assess_portfolio_risk(portfolio, market_data, historical_returns)
    
    print("Risk Assessment Results:")
    print(f"Overall Risk: {assessment.overall_risk.value}")
    print(f"Portfolio VaR (95%): {assessment.portfolio_var:.2%}")
    print(f"Max Drawdown: {abs(assessment.risk_metrics.max_drawdown):.2%}")
    print(f"Sharpe Ratio: {assessment.risk_metrics.sharpe_ratio:.2f}")
    print(f"Warnings: {assessment.warnings}")
    print(f"Recommendations: {assessment.recommendations}")

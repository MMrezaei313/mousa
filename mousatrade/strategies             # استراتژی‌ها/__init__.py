"""
Trading Strategies Module for Mousa Trading Bot
"""

__version__ = '1.0.0'
__author__ = 'Mousa Trading Bot Team'
__description__ = 'Advanced trading strategies implementation'

from .base_strategy import BaseStrategy
from .technical_strategies import (
    MovingAverageCrossover,
    RSIMeanReversion,
    MACDStrategy,
    BollingerBandsStrategy
)
from .ml_strategies import MLStrategy, RandomForestStrategy, XGBoostStrategy
from .portfolio_strategies import PortfolioStrategy, RiskParityStrategy

__all__ = [
    'BaseStrategy',
    'MovingAverageCrossover',
    'RSIMeanReversion', 
    'MACDStrategy',
    'BollingerBandsStrategy',
    'MLStrategy',
    'RandomForestStrategy',
    'XGBoostStrategy',
    'PortfolioStrategy',
    'RiskParityStrategy'
]

"""
Core Trading Engine Module
Heart of the Mousa Trading Bot
"""

__version__ = '1.0.0'
__author__ = 'Mousa Trading Bot Team'

from .trading_engine import TradingEngine
from .state_manager import StateManager
from .event_dispatcher import EventDispatcher
from .performance_tracker import PerformanceTracker

__all__ = [
    'TradingEngine',
    'StateManager', 
    'EventDispatcher',
    'PerformanceTracker'
]

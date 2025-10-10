"""
ثابت‌های全局 MousaTrade
"""

from enum import Enum

# انواع پوزیشن
class PositionType(Enum):
    LONG = "long"
    SHORT = "short" 
    NEUTRAL = "neutral"

# تایم‌فریم‌ها
class Timeframe(Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"

# سطوح ریسک
class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# روندها
class TrendDirection(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

# تنظیمات پیش‌فرض
DEFAULT_CONFIG = {
    'timeframe': '1h',
    'max_analysis_period': 365,
    'risk_free_rate': 0.02,
    'max_drawdown': 0.15,
    'confidence_threshold': 0.7
}

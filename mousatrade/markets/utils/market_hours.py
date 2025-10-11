from typing import Dict, List, Optional
from datetime import datetime, time, timedelta
import pytz
from ..base_market import MarketType, TradingSession

class MarketHoursManager:
    """
    Manage market hours and trading sessions across different markets
    """
    
    def __init__(self):
        self.market_sessions = self._initialize_market_sessions()
    
    def _initialize_market_sessions(self) -> Dict[MarketType, Dict]:
        """Initialize trading sessions for different markets"""
        return {
            MarketType.CRYPTO: {
                'timezone': 'UTC',
                'sessions': {
                    TradingSession._24_7: {
                        'open': time(0, 0),
                        'close': time(23, 59, 59)
                    }
                }
            },
            MarketType.STOCKS: {
                'timezone': 'America/New_York',
                'sessions': {
                    TradingSession.PRE_MARKET: {
                        'open': time(4, 0),
                        'close': time(9, 30)
                    },
                    TradingSession.REGULAR: {
                        'open': time(9, 30),
                        'close': time(16, 0)
                    },
                    TradingSession.AFTER_HOURS: {
                        'open': time(16, 0),
                        'close': time(20, 0)
                    }
                }
            },
            MarketType.FOREX: {
                'timezone': 'UTC',
                'sessions': {
                    'sydney': {'open': time(17, 0), 'close': time(1, 0)},  # GMT+10
                    'tokyo': {'open': time(0, 0), 'close': time(9, 0)},    # GMT+9
                    'london': {'open': time(8, 0), 'close': time(17, 0)},  # GMT+1
                    'new_york': {'open': time(13, 0), 'close': time(22, 0)} # GMT-4
                }
            },
            MarketType.FUTURES: {
                'timezone': 'America/Chicago',
                'sessions': {
                    'regular': {'open': time(8, 30), 'close': time(15, 0)},
                    'electronic': {'open': time(17, 0), 'close': time(16, 0)}  # Next day
                }
            }
        }
    
    def is_market_open(self, market_type: MarketType, current_time: datetime = None) -> bool:
        """Check if a market is currently open"""
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        
        market_info = self.market_sessions.get(market_type)
        if not market_info:
            return False
        
        if market_type == MarketType.CRYPTO:
            return True  # Crypto markets are always open
        
        # Convert current time to market timezone
        market_tz = pytz.timezone(market_info['timezone'])
        market_time = current_time.astimezone(market_tz).time()
        
        # Check if current time falls within any session
        for session_info in market_info['sessions'].values():
            if session_info['open'] <= market_time <= session_info['close']:
                return True
        
        return False
    
    def get_next_market_open(self, market_type: MarketType, current_time: datetime = None) -> datetime:
        """Get next market opening time"""
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        
        market_info = self.market_sessions.get(market_type)
        if not market_info:
            return current_time
        
        if market_type == MarketType.CRYPTO:
            return current_time  # Always open
        
        market_tz = pytz.timezone(market_info['timezone'])
        market_dt = current_time.astimezone(market_tz)
        market_time = market_dt.time()
        
        # Find next session opening
        next_open = None
        for session_info in market_info['sessions'].values():
            if market_time < session_info['open']:
                candidate = market_dt.replace(
                    hour=session_info['open'].hour,
                    minute=session_info['open'].minute,
                    second=0,
                    microsecond=0
                )
                if next_open is None or candidate < next_open:
                    next_open = candidate
            else:
                # Session already passed today, try tomorrow
                candidate = (market_dt + timedelta(days=1)).replace(
                    hour=session_info['open'].hour,
                    minute=session_info['open'].minute,
                    second=0,
                    microsecond=0
                )
                if next_open is None or candidate < next_open:
                    next_open = candidate
        
        return next_open.astimezone(pytz.UTC) if next_open else current_time
    
    def get_current_session(self, market_type: MarketType, current_time: datetime = None) -> Optional[str]:
        """Get current trading session for a market"""
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        
        market_info = self.market_sessions.get(market_type)
        if not market_info:
            return None
        
        if market_type == MarketType.CRYPTO:
            return '24_7'
        
        market_tz = pytz.timezone(market_info['timezone'])
        market_time = current_time.astimezone(market_tz).time()
        
        for session_name, session_info in market_info['sessions'].items():
            if session_info['open'] <= market_time <= session_info['close']:
                return session_name
        
        return None
    
    def get_market_holidays(self, market_type: MarketType, year: int = None) -> List[datetime]:
        """Get market holidays for a specific year"""
        if year is None:
            year = datetime.now().year
        
        # Basic holiday list (US markets)
        holidays = []
        
        if market_type in [MarketType.STOCKS, MarketType.FUTURES]:
            # US Market Holidays 2024
            us_holidays = [
                f"{year}-01-01",  # New Year's Day
                f"{year}-01-15",  # MLK Day
                f"{year}-02-19",  # Presidents Day
                f"{year}-03-29",  # Good Friday
                f"{year}-05-27",  # Memorial Day
                f"{year}-06-19",  # Juneteenth
                f"{year}-07-04",  # Independence Day
                f"{year}-09-02",  # Labor Day
                f"{year}-11-28",  # Thanksgiving
                f"{year}-12-25",  # Christmas
            ]
            
            for date_str in us_holidays:
                holidays.append(datetime.strptime(date_str, "%Y-%m-%date"))
        
        return holidays
    
    def is_market_holiday(self, market_type: MarketType, check_date: datetime = None) -> bool:
        """Check if a date is a market holiday"""
        if check_date is None:
            check_date = datetime.now()
        
        holidays = self.get_market_holidays(market_type, check_date.year)
        return check_date.date() in [h.date() for h in holidays]

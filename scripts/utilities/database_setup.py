import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta
import json

class DatabaseManager:
    def __init__(self, db_path="data/trading_bot.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize all database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market data tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                open_time DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                interval TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, open_time, interval)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_time_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trading signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                strength REAL,
                reasons TEXT,
                price REAL,
                rsi REAL,
                macd REAL,
                timeframe TEXT,
                generated_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Portfolio and positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                position_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                quantity REAL NOT NULL,
                entry_time DATETIME NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL DEFAULT 0,
                pnl_percent REAL DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trade history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                position_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME NOT NULL,
                pnl REAL NOT NULL,
                pnl_percent REAL NOT NULL,
                commission REAL DEFAULT 0,
                strategy TEXT,
                reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Price alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_price REAL NOT NULL,
                condition TEXT NOT NULL,
                is_triggered BOOLEAN DEFAULT FALSE,
                triggered_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                total_return REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                win_rate REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                parameters TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # System configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                description TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database setup completed successfully!")
    
    def initialize_sample_data(self):
        """Initialize database with sample configuration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert default configuration
        default_config = [
            ('trading_enabled', 'false', 'Enable/disable automated trading'),
            ('risk_per_trade', '0.02', 'Risk per trade (2% of portfolio)'),
            ('max_portfolio_risk', '0.1', 'Maximum portfolio risk (10%)'),
            ('default_timeframe', '1h', 'Default trading timeframe'),
            ('api_key', '', 'Exchange API Key'),
            ('api_secret', '', 'Exchange API Secret'),
            ('notification_enabled', 'true', 'Enable/disable notifications')
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO system_config (config_key, config_value, description)
            VALUES (?, ?, ?)
        ''', default_config)
        
        # Insert sample price alerts
        sample_alerts = [
            ('BTCUSDT', 50000, 'ABOVE'),
            ('ETHUSDT', 3000, 'BELOW'),
            ('ADAUSDT', 1.5, 'ABOVE')
        ]
        
        cursor.executemany('''
            INSERT INTO price_alerts (symbol, alert_price, condition)
            VALUES (?, ?, ?)
        ''', sample_alerts)
        
        conn.commit()
        conn.close()
        print("Sample data initialized successfully!")
    
    def get_config_value(self, key, default=None):
        """Get configuration value from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT config_value FROM system_config WHERE config_key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else default
        except Exception as e:
            print(f"Error getting config value: {e}")
            return default
    
    def set_config_value(self, key, value, description=None):
        """Set configuration value in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if description:
                cursor.execute('''
                    INSERT OR REPLACE INTO system_config (config_key, config_value, description)
                    VALUES (?, ?, ?)
                ''', (key, value, description))
            else:
                cursor.execute('''
                    INSERT OR REPLACE INTO system_config (config_key, config_value)
                    VALUES (?, ?)
                ''', (key, value))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error setting config value: {e}")
            return False
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Table row counts
            tables = ['ohlcv_data', 'trading_signals', 'portfolio_positions', 
                     'trade_history', 'price_alerts', 'backtest_results']
            
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[table] = cursor.fetchone()[0]
            
            # Latest data timestamp
            cursor.execute('''
                SELECT MAX(open_time) FROM ohlcv_data
            ''')
            stats['latest_data'] = cursor.fetchone()[0]
            
            # Active positions
            cursor.execute('''
                SELECT COUNT(*) FROM portfolio_positions WHERE is_active = TRUE
            ''')
            stats['active_positions'] = cursor.fetchone()[0]
            
            # Total P&L
            cursor.execute('''
                SELECT SUM(pnl) FROM trade_history
            ''')
            stats['total_pnl'] = cursor.fetchone()[0] or 0
            
            conn.close()
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def backup_database(self, backup_path=None):
        """Create database backup"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backups/trading_bot_backup_{timestamp}.db"
            
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Create backup using SQLite backup API
            source_conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(backup_path)
            
            source_conn.backup(backup_conn)
            
            source_conn.close()
            backup_conn.close()
            
            print(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            print(f"Error creating database backup: {e}")
            return None
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Vacuum database
            cursor.execute('VACUUM')
            
            # Create indexes for better performance
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_data(symbol, open_time)',
                'CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON trading_signals(symbol, generated_at)',
                'CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trade_history(symbol, entry_time)',
                'CREATE INDEX IF NOT EXISTS idx_positions_active ON portfolio_positions(is_active, symbol)'
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            # Analyze database for query optimization
            cursor.execute('ANALYZE')
            
            conn.commit()
            conn.close()
            
            print("Database optimization completed!")
            return True
            
        except Exception as e:
            print(f"Error optimizing database: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep=30):
        """Clean up old data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Delete old OHLCV data
            cursor.execute('''
                DELETE FROM ohlcv_data 
                WHERE open_time < ?
            ''', (cutoff_date,))
            
            ohlcv_deleted = cursor.rowcount
            
            # Delete old real-time data
            cursor.execute('''
                DELETE FROM real_time_data 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            realtime_deleted = cursor.rowcount
            
            # Delete old trading signals
            cursor.execute('''
                DELETE FROM trading_signals 
                WHERE generated_at < ?
            ''', (cutoff_date,))
            
            signals_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"Cleanup completed: {ohlcv_deleted} OHLCV, {realtime_deleted} real-time, {signals_deleted} signal records deleted")
            return {
                'ohlcv_deleted': ohlcv_deleted,
                'realtime_deleted': realtime_deleted,
                'signals_deleted': signals_deleted
            }
            
        except Exception as e:
            print(f"Error cleaning up old data: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    db_manager = DatabaseManager()
    
    # Initialize with sample data
    db_manager.initialize_sample_data()
    
    # Get database statistics
    stats = db_manager.get_database_stats()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Set configuration
    db_manager.set_config_value('trading_enabled', 'true', 'Enable trading')
    
    # Get configuration
    trading_enabled = db_manager.get_config_value('trading_enabled', 'false')
    print(f"Trading enabled: {trading_enabled}")
    
    # Create backup
    backup_path = db_manager.backup_database()
    
    # Optimize database
    db_manager.optimize_database()
    
    # Cleanup old data (keep last 30 days)
    cleanup_stats = db_manager.cleanup_old_data(30)
    print("Cleanup stats:", cleanup_stats)

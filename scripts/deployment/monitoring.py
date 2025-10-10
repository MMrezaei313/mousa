#!/usr/bin/env python3
"""
Mousa Trading Bot - Monitoring Script
Monitors system health, trading performance, and alerts
"""

import psutil
import time
import logging
import smtplib
import sqlite3
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

class SystemMonitor:
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.alert_history = []
        
    def load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration"""
        default_config = {
            "monitoring": {
                "check_interval": 60,  # seconds
                "cpu_threshold": 80,   # percentage
                "memory_threshold": 85, # percentage
                "disk_threshold": 90,  # percentage
                "log_size_threshold": 100,  # MB
                "database_size_threshold": 500  # MB
            },
            "trading": {
                "max_consecutive_losses": 5,
                "max_drawdown_percent": 10,
                "min_win_rate": 40,  # percentage
                "check_interval_minutes": 30
            },
            "alerts": {
                "email_enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "email_from": "",
                "email_to": "",
                "email_password": "",
                "telegram_enabled": False,
                "telegram_bot_token": "",
                "telegram_chat_id": ""
            },
            "apis": {
                "binance_status_url": "https://api.binance.com/api/v3/ping",
                "news_api_url": "https://newsapi.org/v2/top-headlines"
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with default config
                return self.merge_dicts(default_config, user_config)
        except FileNotFoundError:
            logging.warning(f"Monitoring config not found at {config_path}, using defaults")
            return default_config
    
    def merge_dicts(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge dictionaries"""
        result = default.copy()
        for key, value in user.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self.merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def setup_logging(self):
        """Setup monitoring logging"""
        log_dir = Path("logs/monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"monitoring_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("SystemMonitor")
    
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        resources = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters(),
            'running_processes': len(psutil.pids())
        }
        
        # Check for threshold breaches
        alerts = []
        if resources['cpu_percent'] > self.config['monitoring']['cpu_threshold']:
            alerts.append(f"High CPU usage: {resources['cpu_percent']}%")
        
        if resources['memory_percent'] > self.config['monitoring']['memory_threshold']:
            alerts.append(f"High memory usage: {resources['memory_percent']}%")
        
        if resources['disk_percent'] > self.config['monitoring']['disk_threshold']:
            alerts.append(f"High disk usage: {resources['disk_percent']}%")
        
        return {'resources': resources, 'alerts': alerts}
    
    def check_database_health(self) -> Dict:
        """Check database health and performance"""
        db_path = Path("data/trading_bot.db")
        stats = {}
        
        if db_path.exists():
            try:
                # Database size
                db_size_mb = db_path.stat().st_size / (1024 * 1024)
                stats['database_size_mb'] = db_size_mb
                
                if db_size_mb > self.config['monitoring']['database_size_threshold']:
                    stats['alerts'] = [f"Large database size: {db_size_mb:.2f} MB"]
                
                # Database connections and performance
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Table sizes
                cursor.execute("""
                    SELECT name FROM sqlite_master WHERE type='table'
                """)
                tables = cursor.fetchall()
                
                table_stats = {}
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    table_stats[table_name] = row_count
                
                stats['table_counts'] = table_stats
                
                # Check for database errors
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
                stats['integrity_check'] = integrity
                
                if integrity != 'ok':
                    stats['alerts'] = stats.get('alerts', []) + [f"Database integrity issue: {integrity}"]
                
                conn.close()
                
            except Exception as e:
                stats['error'] = f"Database health check failed: {e}"
                stats['alerts'] = [f"Database error: {e}"]
        else:
            stats['error'] = "Database file not found"
            stats['alerts'] = ["Database file not found"]
        
        return stats
    
    def check_trading_performance(self) -> Dict:
        """Check trading performance metrics"""
        performance = {}
        
        try:
            db_path = Path("data/trading_bot.db")
            conn = sqlite3.connect(db_path)
            
            # Recent trading performance
            query = """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl
                FROM trade_history 
                WHERE exit_time >= datetime('now', '-1 day')
            """
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                row = df.iloc[0]
                total_trades = row['total_trades']
                winning_trades = row['winning_trades']
                losing_trades = row['losing_trades']
                
                if total_trades > 0:
                    win_rate = (winning_trades / total_trades) * 100
                    performance['win_rate'] = win_rate
                    performance['total_trades'] = total_trades
                    performance['total_pnl'] = row['total_pnl']
                    
                    # Check for performance issues
                    alerts = []
                    if win_rate < self.config['trading']['min_win_rate']:
                        alerts.append(f"Low win rate: {win_rate:.1f}%")
                    
                    if losing_trades >= self.config['trading']['max_consecutive_losses']:
                        alerts.append(f"Consecutive losses: {losing_trades}")
                    
                    if alerts:
                        performance['alerts'] = alerts
            
            # Current drawdown
            drawdown_query = """
                SELECT 
                    (MAX(capital) - MIN(capital)) / MAX(capital) * 100 as max_drawdown
                FROM (
                    SELECT capital FROM trade_history 
                    WHERE exit_time >= datetime('now', '-7 days')
                    ORDER BY exit_time
                )
            """
            
            drawdown_df = pd.read_sql_query(drawdown_query, conn)
            if not drawdown_df.empty:
                drawdown = drawdown_df.iloc[0]['max_drawdown'] or 0
                performance['max_drawdown'] = drawdown
                
                if drawdown > self.config['trading']['max_drawdown_percent']:
                    performance['alerts'] = performance.get('alerts', []) + [
                        f"High drawdown: {drawdown:.1f}%"
                    ]
            
            conn.close()
            
        except Exception as e:
            performance['error'] = f"Performance check failed: {e}"
        
        return performance
    
    def check_external_apis(self) -> Dict:
        """Check external API status"""
        api_status = {}
        
        # Check Binance API
        try:
            response = requests.get(
                self.config['apis']['binance_status_url'],
                timeout=10
            )
            api_status['binance'] = 'online' if response.status_code == 200 else 'offline'
        except Exception as e:
            api_status['binance'] = 'error'
            api_status['binance_error'] = str(e)
        
        # Check News API (if configured)
        news_api_key = self.config['alerts'].get('news_api_key')
        if news_api_key and news_api_key != "your_news_api_key_here":
            try:
                response = requests.get(
                    self.config['apis']['news_api_url'],
                    params={'apiKey': news_api_key, 'q': 'cryptocurrency', 'pageSize': 1},
                    timeout=10
                )
                api_status['news_api'] = 'online' if response.status_code == 200 else 'offline'
            except Exception as e:
                api_status['news_api'] = 'error'
                api_status['news_api_error'] = str(e)
        
        return api_status
    
    def check_log_files(self) -> Dict:
        """Check log file sizes and health"""
        log_stats = {}
        log_dir = Path("logs")
        
        if log_dir.exists():
            total_size = 0
            large_files = []
            
            for log_file in log_dir.rglob("*.log"):
                size_mb = log_file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                
                if size_mb > self.config['monitoring']['log_size_threshold']:
                    large_files.append({
                        'file': str(log_file),
                        'size_mb': size_mb
                    })
            
            log_stats['total_size_mb'] = total_size
            log_stats['large_files'] = large_files
            
            if large_files:
                log_stats['alerts'] = [f"Large log files detected: {len(large_files)} files"]
        
        return log_stats
    
    def send_alert(self, alert_type: str, message: str, severity: str = "WARNING"):
        """Send alert through configured channels"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alert_history.append(alert)
        self.logger.warning(f"ALERT [{severity}]: {alert_type} - {message}")
        
        # Email alerts
        if self.config['alerts'].get('email_enabled', False):
            self.send_email_alert(alert)
        
        # Telegram alerts
        if self.config['alerts'].get('telegram_enabled', False):
            self.send_telegram_alert(alert)
    
    def send_email_alert(self, alert: Dict):
        """Send alert via email"""
        try:
            smtp_config = self.config['alerts']
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config['email_from']
            msg['To'] = smtp_config['email_to']
            msg['Subject'] = f"[{alert['severity']}] Mousa Trading Bot Alert - {alert['type']}"
            
            body = f"""
            Mousa Trading Bot Alert
            
            Time: {alert['timestamp']}
            Type: {alert['type']}
            Severity: {alert['severity']}
            Message: {alert['message']}
            
            Please check the system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            server.starttls()
            server.login(smtp_config['email_from'], smtp_config['email_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info("Email alert sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def send_telegram_alert(self, alert: Dict):
        """Send alert via Telegram"""
        try:
            telegram_config = self.config['alerts']
            bot_token = telegram_config['telegram_bot_token']
            chat_id = telegram_config['telegram_chat_id']
            
            message = (
                f"🚨 *Mousa Trading Bot Alert*\n\n"
                f"*Type:* {alert['type']}\n"
                f"*Severity:* {alert['severity']}\n"
                f"*Time:* {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"*Message:* {alert['message']}"
            )
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.logger.info("Telegram alert sent successfully")
            else:
                self.logger.error(f"Telegram API error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now(),
            'system_resources': self.check_system_resources(),
            'database_health': self.check_database_health(),
            'trading_performance': self.check_trading_performance(),
            'external_apis': self.check_external_apis(),
            'log_files': self.check_log_files(),
            'active_alerts': len(self.alert_history)
        }
        
        # Collect all alerts
        all_alerts = []
        for section in ['system_resources', 'database_health', 'trading_performance', 'log_files']:
            if 'alerts' in report[section]:
                all_alerts.extend(report[section]['alerts'])
        
        report['all_alerts'] = all_alerts
        
        return report
    
    def save_monitoring_report(self, report: Dict):
        """Save monitoring report to file"""
        try:
            reports_dir = Path("reports/monitoring")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = reports_dir / filename
            
            # Convert datetime objects to strings for JSON serialization
            serializable_report = json.loads(
                json.dumps(report, default=str, indent=2)
            )
            
            with open(filepath, 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            self.logger.info(f"Monitoring report saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save monitoring report: {e}")
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        self.logger.info("Starting continuous monitoring...")
        
        try:
            while True:
                report = self.generate_monitoring_report()
                
                # Send alerts for new issues
                for alert in report['all_alerts']:
                    self.send_alert('System Monitoring', alert, 'WARNING')
                
                # Save report
                self.save_monitoring_report(report)
                
                # Log summary
                self.logger.info(
                    f"Monitoring check completed. "
                    f"Alerts: {len(report['all_alerts'])}, "
                    f"CPU: {report['system_resources']['resources']['cpu_percent']}%, "
                    f"Memory: {report['system_resources']['resources']['memory_percent']}%"
                )
                
                # Wait for next check
                time.sleep(self.config['monitoring']['check_interval'])
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            self.send_alert('Monitoring System', f"Monitoring system error: {e}", 'ERROR')

def main():
    """Main function for command line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mousa Trading Bot Monitoring System')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Run continuous monitoring')
    parser.add_argument('--single-report', '-s', action='store_true',
                       help='Generate single monitoring report')
    parser.add_argument('--config', '-f', default='config/monitoring_config.json',
                       help='Path to monitoring configuration file')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.config)
    
    if args.continuous:
        monitor.run_continuous_monitoring()
    elif args.single_report:
        report = monitor.generate_monitoring_report()
        monitor.save_monitoring_report(report)
        print("Single monitoring report generated")
    else:
        print("Please specify either --continuous or --single-report")

if __name__ == '__main__':
    main()

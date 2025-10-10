import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from technical_analysis import TechnicalAnalyzer
from signal_generator import TradingSignalGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    def __init__(self, initial_capital=10000, commission=0.001, db_path="data/market_data.db"):
        self.initial_capital = initial_capital
        self.commission = commission
        self.db_path = db_path
        self.analyzer = TechnicalAnalyzer()
        self.signal_generator = TradingSignalGenerator(db_path)
        
    def run_backtest(self, symbol: str, timeframe: str, 
                    start_date: str, end_date: str,
                    strategy_type: str = 'technical') -> Dict:
        """Run comprehensive backtest for a strategy"""
        
        # Get historical data
        df = self._get_historical_data(symbol, timeframe, start_date, end_date)
        if df is None or df.empty:
            return {"error": "No data available for backtest"}
        
        # Calculate technical indicators
        df = self.analyzer.calculate_all_indicators(df)
        
        # Generate trading signals based on strategy
        if strategy_type == 'technical':
            signals = self._generate_technical_signals(df)
        elif strategy_type == 'momentum':
            signals = self._generate_momentum_signals(df)
        elif strategy_type == 'mean_reversion':
            signals = self._generate_mean_reversion_signals(df)
        else:
            signals = self._generate_technical_signals(df)
        
        # Execute backtest
        results = self._execute_trades(df, signals)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(results, df)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'period': f"{start_date} to {end_date}",
            'strategy': strategy_type,
            'results': results,
            'performance': performance,
            'equity_curve': self._generate_equity_curve(results)
        }
    
    def _get_historical_data(self, symbol: str, timeframe: str, 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve historical data for backtesting"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT open_time, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE symbol = ? AND interval = ? AND open_time BETWEEN ? AND ?
                ORDER BY open_time ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=[symbol, timeframe, start_date, end_date])
            conn.close()
            
            if not df.empty:
                df['open_time'] = pd.to_datetime(df['open_time'])
                df.set_index('open_time', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error retrieving historical data: {e}")
            return None
    
    def _generate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using technical analysis"""
        signals_df = self.analyzer.generate_signals(df)
        
        # Convert to trading positions
        positions = []
        current_position = 0  # 0: out, 1: long, -1: short
        
        for i, signal in signals_df.iterrows():
            signal_type = signal['signal']
            
            if signal_type in ['STRONG_BUY', 'BUY'] and current_position <= 0:
                positions.append(1)  # Enter long
                current_position = 1
            elif signal_type in ['STRONG_SELL', 'SELL'] and current_position >= 0:
                positions.append(-1)  # Enter short
                current_position = -1
            else:
                positions.append(current_position)  # Hold current position
        
        signals_df['position'] = positions
        return signals_df
    
    def _generate_momentum_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based trading signals"""
        signals = []
        
        # RSI Momentum
        df['rsi_signal'] = np.where(df['rsi_14'] < 30, 1, 
                                  np.where(df['rsi_14'] > 70, -1, 0))
        
        # MACD Momentum
        df['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Combined momentum signal
        for i in range(len(df)):
            rsi_sig = df['rsi_signal'].iloc[i] if 'rsi_signal' in df.columns else 0
            macd_sig = df['macd_signal'].iloc[i] if 'macd_signal' in df.columns else 0
            
            # Simple majority voting
            total_signal = rsi_sig + macd_sig
            
            if total_signal >= 1:
                signals.append(1)  # Long
            elif total_signal <= -1:
                signals.append(-1)  # Short
            else:
                signals.append(0)  # Neutral
        
        result_df = pd.DataFrame({
            'timestamp': df.index,
            'signal': ['BUY' if x == 1 else 'SELL' if x == -1 else 'HOLD' for x in signals],
            'position': signals
        })
        
        return result_df
    
    def _generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion trading signals"""
        signals = []
        
        # Bollinger Bands mean reversion
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            for i in range(len(df)):
                price = df['close'].iloc[i]
                bb_upper = df['bb_upper'].iloc[i]
                bb_lower = df['bb_lower'].iloc[i]
                
                if price > bb_upper:
                    signals.append(-1)  # Short (overbought)
                elif price < bb_lower:
                    signals.append(1)   # Long (oversold)
                else:
                    signals.append(0)   # Neutral
        else:
            signals = [0] * len(df)
        
        result_df = pd.DataFrame({
            'timestamp': df.index,
            'signal': ['BUY' if x == 1 else 'SELL' if x == -1 else 'HOLD' for x in signals],
            'position': signals
        })
        
        return result_df
    
    def _execute_trades(self, df: pd.DataFrame, signals: pd.DataFrame) -> List[Dict]:
        """Execute trades based on signals"""
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        capital = self.initial_capital
        equity_curve = [capital]
        
        for i in range(len(signals)):
            current_time = signals.iloc[i]['timestamp']
            current_price = df.loc[current_time, 'close'] if current_time in df.index else df.iloc[i]['close']
            signal_position = signals.iloc[i]['position']
            
            # Close position if signal changes
            if position != 0 and signal_position != position:
                # Calculate P&L
                if position == 1:  # Long position
                    pnl = (current_price - entry_price) / entry_price
                else:  # Short position
                    pnl = (entry_price - current_price) / entry_price
                
                # Apply commission
                pnl_after_commission = pnl - (2 * self.commission)
                
                # Update capital
                capital *= (1 + pnl_after_commission)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': 'LONG' if position == 1 else 'SHORT',
                    'pnl': pnl_after_commission,
                    'capital': capital
                })
                
                position = 0
                entry_price = 0
                entry_time = None
            
            # Open new position
            if position == 0 and signal_position != 0:
                position = signal_position
                entry_price = current_price
                entry_time = current_time
            
            equity_curve.append(capital)
        
        # Close any open position at the end
        if position != 0:
            current_price = df.iloc[-1]['close']
            if position == 1:  # Long
                pnl = (current_price - entry_price) / entry_price
            else:  # Short
                pnl = (entry_price - current_price) / entry_price
            
            pnl_after_commission = pnl - (2 * self.commission)
            capital *= (1 + pnl_after_commission)
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': 'LONG' if position == 1 else 'SHORT',
                'pnl': pnl_after_commission,
                'capital': capital
            })
        
        return trades
    
    def _calculate_performance_metrics(self, trades: List[Dict], df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0
            }
        
        # Convert trades to DataFrame for easier calculation
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_return = (trades_df['capital'].iloc[-1] / self.initial_capital - 1) * 100
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl'].tolist()
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Maximum drawdown
        equity_curve = [self.initial_capital] + trades_df['capital'].tolist()
        running_max = pd.Series(equity_curve).cummax()
        drawdowns = (pd.Series(equity_curve) - running_max) / running_max
        max_drawdown = drawdowns.min() * 100
        
        # Average trade metrics
        avg_winning_trade = trades_df[trades_df['pnl'] > 0]['pnl'].mean() * 100 if winning_trades > 0 else 0
        avg_losing_trade = trades_df[trades_df['pnl'] < 0]['pnl'].mean() * 100 if losing_trades > 0 else 0
        avg_trade = trades_df['pnl'].mean() * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'avg_trade': avg_trade,
            'final_capital': trades_df['capital'].iloc[-1] if not trades_df.empty else self.initial_capital
        }
    
    def _generate_equity_curve(self, trades: List[Dict]) -> List[Dict]:
        """Generate equity curve data for plotting"""
        equity_data = []
        capital = self.initial_capital
        
        equity_data.append({
            'timestamp': datetime.now() - timedelta(days=1),
            'equity': capital
        })
        
        for trade in trades:
            capital = trade['capital']
            equity_data.append({
                'timestamp': trade['exit_time'],
                'equity': capital
            })
        
        return equity_data
    
    def run_comparative_analysis(self, symbol: str, timeframes: List[str], 
                               strategies: List[str], start_date: str, end_date: str) -> Dict:
        """Run comparative analysis across different strategies and timeframes"""
        results = {}
        
        for strategy in strategies:
            strategy_results = {}
            for timeframe in timeframes:
                print(f"Testing {strategy} strategy on {timeframe} timeframe...")
                result = self.run_backtest(symbol, timeframe, start_date, end_date, strategy)
                strategy_results[timeframe] = result['performance']
            
            results[strategy] = strategy_results
        
        return {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'comparative_results': results,
            'best_strategy': self._find_best_strategy(results)
        }
    
    def _find_best_strategy(self, results: Dict) -> Dict:
        """Find the best performing strategy"""
        best_return = -float('inf')
        best_strategy = None
        best_timeframe = None
        
        for strategy, timeframes in results.items():
            for timeframe, performance in timeframes.items():
                if performance['total_return'] > best_return:
                    best_return = performance['total_return']
                    best_strategy = strategy
                    best_timeframe = timeframe
        
        return {
            'strategy': best_strategy,
            'timeframe': best_timeframe,
            'return': best_return
        }
    
    def plot_backtest_results(self, backtest_results: Dict, save_path: str = None):
        """Plot backtest results"""
        trades_df = pd.DataFrame(backtest_results['results'])
        
        if trades_df.empty:
            print("No trades to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        equity_curve = backtest_results['equity_curve']
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        ax1.plot(equity_df.index, equity_df['equity'], label='Equity Curve', linewidth=2)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Daily returns
        trades_df['daily_return'] = trades_df['pnl'] * 100
        ax2.hist(trades_df['daily_return'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Daily Returns')
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
        
        # Drawdown
        equity_series = pd.Series([x['equity'] for x in equity_curve])
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        
        ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax3.plot(drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Trade Number')
        ax3.grid(True)
        
        # Performance metrics table
        metrics = backtest_results['performance']
        metric_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        metric_values = [
            f"{metrics['total_return']:.2f}%",
            f"{metrics['sharpe_ratio']:.2f}",
            f"{metrics['max_drawdown']:.2f}%",
            f"{metrics['win_rate']:.2f}%"
        ]
        
        ax4.axis('off')
        table = ax4.table(
            cellText=[metric_values],
            colLabels=metric_names,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Performance Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    backtester = BacktestEngine(initial_capital=10000, commission=0.001)
    
    # Run single backtest
    results = backtester.run_backtest(
        symbol='BTCUSDT',
        timeframe='1h',
        start_date='2024-01-01',
        end_date='2024-03-01',
        strategy_type='technical'
    )
    
    print("Backtest Results:")
    print(f"Total Return: {results['performance']['total_return']:.2f}%")
    print(f"Win Rate: {results['performance']['win_rate']:.2f}%")
    print(f"Total Trades: {results['performance']['total_trades']}")
    
    # Run comparative analysis
    comparative = backtester.run_comparative_analysis(
        symbol='BTCUSDT',
        timeframes=['1h', '4h', '1d'],
        strategies=['technical', 'momentum', 'mean_reversion'],
        start_date='2024-01-01',
        end_date='2024-03-01'
    )
    
    print("\nBest Strategy:")
    print(comparative['best_strategy'])
    
    # Plot results
    backtester.plot_backtest_results(results)

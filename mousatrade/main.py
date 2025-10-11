#!/usr/bin/env python3
"""
Mousa Trading System - Main Entry Point
Enhanced with Multi-Market Support and Resiliency
"""

import asyncio
import logging
from typing import Dict, List, Optional

# Import Mousa core modules
from mousatrade.core.trader import MousaTrader
from mousatrade.core.config import Config
from mousatrade.core.logger import setup_logging

# Import new multi-market modules
from mousatrade.markets import MarketRegistry, BinanceMarket, RobinhoodMarket, ForexMarket
from mousatrade.markets import StockAnalyzer, CryptoAnalyzer, CorrelationTracker
from mousatrade.resiliency import resilient_trade_execution, ErrorHandler, get_config

class EnhancedMousaTrader:
    """
    Enhanced version of MousaTrader with multi-market support
    """
    
    def __init__(self, config_path: str = "config.json"):
        # Load original Mousa configuration
        self.config = Config(config_path)
        
        # Initialize original Mousa trader
        self.original_trader = MousaTrader(config_path)
        
        # Initialize multi-market system
        self.market_registry = MarketRegistry()
        self.error_handler = ErrorHandler()
        self.resiliency_config = get_config("production")
        
        # Initialize analyzers
        self.stock_analyzer = StockAnalyzer()
        self.crypto_analyzer = CryptoAnalyzer()
        self.correlation_tracker = CorrelationTracker()
        
        # Trading state
        self.is_running = False
        self.active_strategies = {}
    
    def setup_markets(self):
        """Setup and connect to multiple markets"""
        try:
            print("🚀 Setting up multi-market trading system...")
            
            # Initialize markets based on configuration
            if self.config.get('exchanges.binance.enabled', False):
                binance = BinanceMarket(
                    api_key=self.config.get('exchanges.binance.api_key'),
                    secret=self.config.get('exchanges.binance.api_secret')
                )
                self.market_registry.register_market("binance", binance)
            
            if self.config.get('exchanges.robinhood.enabled', False):
                robinhood = RobinhoodMarket(
                    username=self.config.get('exchanges.robinhood.username'),
                    password=self.config.get('exchanges.robinhood.password')
                )
                self.market_registry.register_market("robinhood", robinhood)
            
            if self.config.get('exchanges.forex.enabled', False):
                forex = ForexMarket(
                    broker=self.config.get('exchanges.forex.broker', 'oanda'),
                    api_key=self.config.get('exchanges.forex.api_key')
                )
                self.market_registry.register_market("forex", forex)
            
            # Connect to all registered markets
            connection_results = self.market_registry.connect_all()
            print("✅ Market connections established:", connection_results)
            
            return True
            
        except Exception as e:
            self.error_handler.handle_trading_error(e, {"context": "market_setup"})
            return False
    
    async def start_trading(self):
        """Start the enhanced trading system"""
        if self.is_running:
            print("⚠️ Trading system is already running")
            return
        
        print("🎯 Starting Enhanced Mousa Trading System...")
        
        # Setup markets
        if not self.setup_markets():
            print("❌ Failed to setup markets")
            return
        
        self.is_running = True
        
        # Start original Mousa strategies
        await self.original_trader.start()
        
        # Start multi-market monitoring
        asyncio.create_task(self.multi_market_monitor())
        
        # Start correlation analysis
        asyncio.create_task(self.correlation_analysis_loop())
        
        print("✅ Enhanced Mousa Trading System started successfully")
    
    async def stop_trading(self):
        """Stop the trading system"""
        if not self.is_running:
            return
        
        print("🛑 Stopping Enhanced Mousa Trading System...")
        
        self.is_running = False
        
        # Stop original Mousa trader
        await self.original_trader.stop()
        
        # Disconnect from all markets
        self.market_registry.disconnect_all()
        
        print("✅ Enhanced Mousa Trading System stopped")
    
    async def multi_market_monitor(self):
        """Monitor multiple markets for opportunities"""
        while self.is_running:
            try:
                await self.check_arbitrage_opportunities()
                await self.check_market_correlations()
                await self.update_portfolio_analysis()
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                self.error_handler.handle_trading_error(e, {"context": "market_monitor"})
                await asyncio.sleep(60)  # Wait longer on error
    
    async def check_arbitrage_opportunities(self):
        """Check for arbitrage opportunities across markets"""
        try:
            # Major symbols to check for arbitrage
            symbols = ['BTC/USDT', 'ETH/USDT', 'AAPL', 'SPY']
            
            for symbol in symbols:
                best_buy = self.market_registry.get_best_price(symbol, 'buy')
                best_sell = self.market_registry.get_best_price(symbol, 'sell')
                
                if best_buy['price'] and best_sell['price']:
                    spread = best_sell['price'] - best_buy['price']
                    spread_percent = (spread / best_buy['price']) * 100
                    
                    if spread_percent > 0.5:  # 0.5% threshold
                        print(f"💰 Arbitrage Opportunity: {symbol}")
                        print(f"   Buy from: {best_buy['market']} @ {best_buy['price']}")
                        print(f"   Sell to: {best_sell['market']} @ {best_sell['price']}")
                        print(f"   Spread: {spread_percent:.2f}%")
                        
        except Exception as e:
            self.error_handler.handle_trading_error(e, {"context": "arbitrage_check"})
    
    async def correlation_analysis_loop(self):
        """Continuous correlation analysis"""
        while self.is_running:
            try:
                # Update price data for correlation tracking
                symbols = ['BTC/USDT', 'ETH/USDT', 'AAPL', 'SPY', 'EUR/USD']
                
                for symbol in symbols:
                    for market_name in self.market_registry.get_connected_markets():
                        market = self.market_registry.get_market(market_name)
                        if market and symbol in market.get_symbols():
                            ohlcv = market.get_ohlcv(symbol, '1d', 30)
                            if not ohlcv.empty:
                                self.correlation_tracker.update_prices(symbol, ohlcv['close'])
                
                # Calculate and log correlations
                corr_matrix = self.correlation_tracker.calculate_correlation_matrix()
                if not corr_matrix.empty:
                    high_corr = self.correlation_tracker.find_highly_correlated_pairs(0.7)
                    if high_corr:
                        print("🔗 Highly Correlated Pairs:")
                        for pair in high_corr[:5]:  # Top 5
                            print(f"   {pair[0]} - {pair[1]}: {pair[2]:.3f}")
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.error_handler.handle_trading_error(e, {"context": "correlation_analysis"})
                await asyncio.sleep(60)
    
    async def update_portfolio_analysis(self):
        """Update portfolio analysis across all markets"""
        try:
            balances = self.market_registry.get_balances()
            total_value = 0
            portfolio = {}
            
            # Calculate total portfolio value
            for market_name, market_balances in balances.items():
                for currency, amount in market_balances.items():
                    if currency not in portfolio:
                        portfolio[currency] = 0
                    portfolio[currency] += amount
            
            print("📊 Multi-Market Portfolio Summary:")
            for currency, amount in portfolio.items():
                if amount > 0:
                    print(f"   {currency}: {amount:,.2f}")
            
        except Exception as e:
            self.error_handler.handle_trading_error(e, {"context": "portfolio_analysis"})
    
    @resilient_trade_execution(exchange_name="multi_market")
    def execute_smart_trade(self, symbol: str, action: str, amount: float, 
                           strategy: str = "best_price"):
        """
        Execute trade using smart order routing across multiple markets
        """
        try:
            if strategy == "best_price":
                # Find best market for this trade
                best_market = self.market_registry.get_best_price(symbol, action)
                
                if best_market['market'] and best_market['price']:
                    market = self.market_registry.get_market(best_market['market'])
                    if market:
                        result = market.place_order(symbol, 'MARKET', action, amount)
                        
                        print(f"🎯 Smart Trade Executed:")
                        print(f"   Symbol: {symbol}")
                        print(f"   Action: {action}")
                        print(f"   Amount: {amount}")
                        print(f"   Market: {best_market['market']}")
                        print(f"   Price: {best_market['price']}")
                        
                        return result
            
            # Fallback: use original Mousa trader
            return self.original_trader.execute_trade(symbol, action, amount)
            
        except Exception as e:
            self.error_handler.handle_trading_error(e, {
                "context": "smart_trade",
                "symbol": symbol,
                "action": action,
                "amount": amount
            })
            raise
    
    def get_market_health(self):
        """Get health status of all connected markets"""
        connected_markets = self.market_registry.get_connected_markets()
        health_status = {
            'connected_markets': connected_markets,
            'total_markets': len(self.market_registry.markets),
            'balances': self.market_registry.get_balances()
        }
        
        return health_status

async def main():
    """Main entry point for Enhanced Mousa Trading System"""
    
    # Setup logging
    setup_logging()
    logging.info("Starting Enhanced Mousa Trading System")
    
    # Initialize enhanced trader
    trader = EnhancedMousaTrader("config.json")
    
    try:
        # Start trading
        await trader.start_trading()
        
        # Keep the system running
        print("\n📍 Enhanced Mousa is running. Press Ctrl+C to stop.")
        
        # Example: Execute a sample trade after 10 seconds
        await asyncio.sleep(10)
        
        # Sample smart trade (optional - for testing)
        if trader.config.get('trading.demo_mode', True):
            print("\n🧪 Executing demo trade...")
            try:
                result = trader.execute_smart_trade("BTC/USDT", "buy", 0.001)
                print(f"Demo trade result: {result}")
            except Exception as e:
                print(f"Demo trade failed: {e}")
        
        # Wait indefinitely (or until stop signal)
        while trader.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Received stop signal...")
    except Exception as e:
        logging.error(f"System error: {e}")
        trader.error_handler.handle_trading_error(e, {"context": "main_loop"})
    finally:
        # Clean shutdown
        await trader.stop_trading()
        logging.info("Enhanced Mousa Trading System stopped")

if __name__ == "__main__":
    # Run the enhanced trading system
    asyncio.run(main())

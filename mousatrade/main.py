#!/usr/bin/env python3
"""
Mousa Trading System - Main Entry Point
Enhanced with Multi-Market Support and Resiliency
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

# Import configuration
from config import get_config, init_config

# Import Mousa core modules
from mousatrade.core.trader import MousaTrader
from mousatrade.core.logger import setup_logging

# Import new multi-market modules
from mousatrade.markets import MarketRegistry, BinanceMarket, RobinhoodMarket, ForexMarket, KuCoinMarket
from mousatrade.markets import StockAnalyzer, CryptoAnalyzer, CorrelationTracker
from mousatrade.resiliency import resilient_trade_execution, ErrorHandler

class EnhancedMousaTrader:
    """
    Enhanced version of MousaTrader with multi-market support
    """
    
    def __init__(self, config_path: str = "config.json"):
        # Initialize configuration
        self.config = init_config(config_path)
        
        # Initialize original Mousa trader
        self.original_trader = MousaTrader(config_path)
        
        # Initialize multi-market system
        self.market_registry = MarketRegistry()
        self.error_handler = ErrorHandler()
        
        # Initialize analyzers
        self.stock_analyzer = StockAnalyzer()
        self.crypto_analyzer = CryptoAnalyzer()
        self.correlation_tracker = CorrelationTracker()
        
        # Trading state
        self.is_running = False
        self.active_strategies = {}

    def setup_markets(self):
        """Setup and connect to multiple markets based on config"""
        try:
            print("🚀 Setting up multi-market trading system...")
            
            # Get enabled exchanges from config
            enabled_exchanges = self.config.get_enabled_exchanges()
            print(f"📊 Enabled exchanges: {enabled_exchanges}")
            
            # Initialize markets based on configuration
            if "binance" in enabled_exchanges:
                binance_config = self.config.get_exchange_config("binance")
                binance = BinanceMarket(
                    api_key=binance_config.get("api_key"),
                    secret=binance_config.get("api_secret"),
                    sandbox=binance_config.get("sandbox", False)
                )
                self.market_registry.register_market("binance", binance)
                print("✅ Binance market registered")
            
            if "robinhood" in enabled_exchanges:
                robinhood_config = self.config.get_exchange_config("robinhood")
                robinhood = RobinhoodMarket(
                    username=robinhood_config.get("username"),
                    password=robinhood_config.get("password")
                )
                self.market_registry.register_market("robinhood", robinhood)
                print("✅ Robinhood market registered")
            
            if "forex" in enabled_exchanges:
                forex_config = self.config.get_exchange_config("forex")
                forex = ForexMarket(
                    broker=forex_config.get("broker", "oanda"),
                    api_key=forex_config.get("api_key"),
                    account_id=forex_config.get("account_id")
                )
                self.market_registry.register_market("forex", forex)
                print("✅ Forex market registered")
            
            if "kucoin" in enabled_exchanges:
                kucoin_config = self.config.get_exchange_config("kucoin")
                kucoin = KuCoinMarket(
                    api_key=kucoin_config.get("api_key"),
                    secret=kucoin_config.get("api_secret"),
                    password=kucoin_config.get("api_passphrase")
                )
                self.market_registry.register_market("kucoin", kucoin)
                print("✅ KuCoin market registered")
            
            # Connect to all registered markets
            connection_results = self.market_registry.connect_all()
            print("✅ Market connections established:", connection_results)
            
            return True
            
        except Exception as e:
            self.error_handler.handle_trading_error(e, {"context": "market_setup"})
            return False

    # بقیه متدها مانند قبل...
    # [کدهای قبلی که فرستادم رو اینجا قرار بده]

async def main():
    """Main entry point for Enhanced Mousa Trading System"""
    
    # Initialize configuration
    config = init_config()
    
    # Validate config
    if not config.validate_config():
        print("❌ Configuration validation failed. Please check your config.")
        return
    
    # Setup logging
    log_level = config.get("mousa.log_level", "INFO")
    setup_logging(level=log_level)
    
    logging.info("Starting Enhanced Mousa Trading System")
    
    # Initialize enhanced trader
    trader = EnhancedMousaTrader()
    
    try:
        # Start trading
        await trader.start_trading()
        
        # Keep the system running
        print("\n📍 Enhanced Mousa is running. Press Ctrl+C to stop.")
        print("💡 Check logs for trading activity and market analysis")
        
        # Wait indefinitely
        while trader.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Received stop signal...")
    except Exception as e:
        logging.error(f"System error: {e}")
        trader.error_handler.handle_trading_error(e, {"context": "main_loop"})
    finally:
        await trader.stop_trading()
        logging.info("Enhanced Mousa Trading System stopped")

if __name__ == "__main__":
    asyncio.run(main())

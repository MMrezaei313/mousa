#!/usr/bin/env python3
"""
MousaTrade Advisor - مشاور هوشمند ترید
مشاور شما برای تحلیل بازار - نه ربات معامله‌گر!
"""

import logging
import asyncio
from typing import Dict, List, Optional
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta

from mousatrade.advisor.position_advisor import PositionAdvisor
from mousatrade.advisor.strategy_advisor import StrategyAdvisor
from mousatrade.analysis.technical import TechnicalAnalyzer
from mousatrade.analysis.fundamental import FundamentalAnalyzer
from mousatrade.analysis.sentiment import SentimentAnalyzer
from mousatrade.data.dataprovider import DataProvider
from mousatrade.brokers import BrokerFactory
from mousatrade.backtesting.backtesting import BacktestingEngine
from mousatrade.persistence.models import AnalysisResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MousaTradeAdvisor:
    """کلاس اصلی مشاور MousaTrade"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize core components
        self.data_provider = DataProvider()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.position_advisor = PositionAdvisor()
        self.strategy_advisor = StrategyAdvisor()
        self.backtesting_engine = BacktestingEngine()
        self.broker_factory = BrokerFactory()
        
        self.current_broker = None
        self.analysis_history = []
        
        logger.info("🎯 MousaTrade Advisor راه‌اندازی شد")
        logger.info("📊 شامل تمام ابزارهای تحلیل FreqTrade + تحلیل فاندامنتال")
    
    def set_broker(self, broker_name: str, api_key: str = None, secret: str = None) -> bool:
        """تنظیم بروکر برای دریافت داده‌های بازار"""
        try:
            self.current_broker = self.broker_factory.create_broker(
                broker_name, api_key, secret
            )
            logger.info(f"✅ بروکر {broker_name} تنظیم شد")
            return True
        except Exception as e:
            logger.error(f"❌ خطا در تنظیم بروکر {broker_name}: {e}")
            return False
    
    def get_comprehensive_analysis(self, symbol: str, timeframe: str = "1h") -> Dict:
        """آنالیز کامل برای یک نماد"""
        if not self.current_broker:
            return {"error": "لطفاً ابتدا بروکر را تنظیم کنید"}
        
        try:
            # دریافت داده‌های تاریخی
            historical_data = self.current_broker.get_historical_data(
                symbol, timeframe, days=30
            )
            
            # تحلیل تکنیکال پیشرفته
            technical_analysis = self.technical_analyzer.comprehensive_analysis(
                historical_data
            )
            
            # تحلیل فاندامنتال
            fundamental_analysis = self.fundamental_analyzer.analyze(symbol)
            
            # تحلیل احساسات
            sentiment_analysis = self.sentiment_analyzer.analyze(symbol)
            
            # دریافت پیشنهاد پوزیشن
            position_advice = self.position_advisor.get_position_advice(
                symbol, historical_data, technical_analysis, 
                fundamental_analysis, sentiment_analysis
            )
            
            # پیشنهاد استراتژی
            strategy_advice = self.strategy_advisor.get_strategy_recommendation(
                symbol, technical_analysis, position_advice
            )
            
            # ساخت نتیجه نهایی
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "position_advice": position_advice,
                "technical_analysis": technical_analysis,
                "fundamental_analysis": fundamental_analysis,
                "sentiment_analysis": sentiment_analysis,
                "strategy_recommendation": strategy_advice,
                "risk_assessment": self._calculate_risk_assessment(
                    position_advice, technical_analysis
                )
            }
            
            # ذخیره در تاریخچه
            self._save_analysis_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"خطا در تحلیل {symbol}: {e}")
            return {"error": f"خطا در تحلیل: {str(e)}"}
    
    def backtest_strategy(self, symbol: str, strategy_name: str, 
                         timeframe: str = "1h", days: int = 90) -> Dict:
        """بک‌تست استراتژی روی داده‌های تاریخی"""
        try:
            historical_data = self.current_broker.get_historical_data(
                symbol, timeframe, days=days
            )
            
            backtest_result = self.backtesting_engine.run_backtest(
                strategy_name, historical_data, symbol
            )
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"خطا در بک‌تست {symbol}: {e}")
            return {"error": f"خطا در بک‌تست: {str(e)}"}
    
    def get_market_overview(self, symbols: List[str]) -> Dict:
        """بررسی کلی بازار برای چند نماد"""
        overview = {}
        
        for symbol in symbols:
            try:
                analysis = self.get_comprehensive_analysis(symbol)
                overview[symbol] = {
                    "position": analysis["position_advice"]["position_type"],
                    "confidence": analysis["position_advice"]["confidence"],
                    "trend": analysis["technical_analysis"]["trend"],
                    "risk": analysis["risk_assessment"]["level"]
                }
            except Exception as e:
                overview[symbol] = {"error": str(e)}
        
        return overview
    
    def _calculate_risk_assessment(self, position_advice: Dict, 
                                 technical_analysis: Dict) -> Dict:
        """محاسبه سطح ریسک"""
        confidence = position_advice.get("confidence", 0)
        volatility = technical_analysis.get("volatility", 0)
        trend_strength = technical_analysis.get("trend_strength", 0)
        
        # محاسبه امتیاز ریسک
        risk_score = (1 - confidence/100) * 0.6 + volatility * 0.4
        
        if risk_score < 0.3:
            level = "کم"
            color = "green"
        elif risk_score < 0.6:
            level = "متوسط"
            color = "orange"
        else:
            level = "بالا"
            color = "red"
        
        return {
            "level": level,
            "score": round(risk_score, 2),
            "color": color,
            "factors": {
                "confidence_impact": round(1 - confidence/100, 2),
                "volatility_impact": round(volatility, 2)
            }
        }
    
    def _save_analysis_result(self, result: Dict):
        """ذخیره نتیجه تحلیل در تاریخچه"""
        analysis_record = AnalysisResult(
            symbol=result["symbol"],
            timeframe=result["timeframe"],
            position_type=result["position_advice"]["position_type"],
            confidence=result["position_advice"]["confidence"],
            technical_data=result["technical_analysis"],
            timestamp=datetime.now()
        )
        
        self.analysis_history.append(analysis_record)
        
        # نگه‌داری فقط 100 تحلیل اخیر
        if len(self.analysis_history) > 100:
            self.analysis_history.pop(0)

# ایجاد نمونه اصلی
mousatrade_advisor = MousaTradeAdvisor()

# راه‌اندازی سرور وب
app = Flask(__name__)
app.secret_key = 'mousatrade-secret-key-2024'

@app.route('/')
def index():
    """صفحه اصلی"""
    return render_template('index.html')

@app.route('/api/set_broker', methods=['POST'])
def api_set_broker():
    """تنظیم بروکر از طریق API"""
    data = request.get_json()
    broker_name = data.get('broker')
    
    success = mousatrade_advisor.set_broker(broker_name)
    
    if success:
        session['broker'] = broker_name
        return jsonify({
            "status": "success",
            "message": f"بروکر {broker_name} با موفقیت تنظیم شد"
        })
    else:
        return jsonify({
            "status": "error",
            "message": f"خطا در تنظیم بروکر {broker_name}"
        }), 400

@app.route('/api/analyze/<symbol>')
def api_analyze(symbol):
    """دریافت تحلیل کامل برای یک نماد"""
    timeframe = request.args.get('timeframe', '1h')
    
    analysis = mousatrade_advisor.get_comprehensive_analysis(symbol, timeframe)
    
    return jsonify(analysis)

@app.route('/api/backtest/<symbol>')
def api_backtest(symbol):
    """بک‌تست استراتژی"""
    strategy = request.args.get('strategy', 'default')
    timeframe = request.args.get('timeframe', '1h')
    days = int(request.args.get('days', '90'))
    
    result = mousatrade_advisor.backtest_strategy(symbol, strategy, timeframe, days)
    
    return jsonify(result)

@app.route('/api/market-overview')
def api_market_overview():
    """بررسی کلی بازار"""
    symbols_param = request.args.get('symbols', 'BTC/USDT,ETH/USDT')
    symbols = [s.strip() for s in symbols_param.split(',')]
    
    overview = mousatrade_advisor.get_market_overview(symbols)
    
    return jsonify(overview)

if __name__ == "__main__":
    logger.info("🚀 در حال راه‌اندازی سرور MousaTrade...")
    app.run(host="0.0.0.0", port=5000, debug=True)

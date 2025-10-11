"""
Route definitions for Mousa Trading Bot web interface
"""

from flask import Blueprint, render_template, jsonify, request, session, redirect, url_for, flash
from .auth import login_required, admin_required, auth
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, List
import os
from pathlib import Path

# Create blueprints
main = Blueprint('main', __name__)
api = Blueprint('api', __name__, url_prefix='/api')

logger = logging.getLogger(__name__)

# Main routes
@main.route('/')
def index():
    """Home page"""
    if 'user_id' in session:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')

@main.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@main.route('/strategies')
@login_required
def strategies():
    """Trading strategies management"""
    return render_template('strategies.html')

@main.route('/optimization')
@login_required
def optimization():
    """Strategy optimization"""
    return render_template('optimization.html')

@main.route('/backtesting')
@login_required
def backtesting():
    """Backtesting interface"""
    return render_template('backtesting.html')

@main.route('/portfolio')
@login_required
def portfolio():
    """Portfolio management"""
    return render_template('portfolio.html')

@main.route('/settings')
@login_required
def settings():
    """Application settings"""
    return render_template('settings.html')

@main.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = auth.authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            
            flash(f'Welcome back, {username}!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
        elif auth.create_user(username, email, password):
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('main.login'))
        else:
            flash('Registration failed. Username or email may already exist.', 'error')
    
    return render_template('register.html')

@main.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('main.index'))

# API Routes
@api.route('/health')
def health():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@api.route('/auth/login', methods=['POST'])
def api_login():
    """API login endpoint"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = auth.authenticate_user(username, password)
    if user:
        token = auth.generate_token(user['id'], user['username'], user['role'])
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'role': user['role']
            }
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Invalid credentials'
        }), 401

@api.route('/auth/verify')
@login_required
def api_verify():
    """Verify authentication token"""
    return jsonify({
        'success': True,
        'user': request.user
    })

@api.route('/strategies/list')
@login_required
def list_strategies():
    """Get list of available strategies"""
    try:
        from mousatrade.strategies import create_strategy, create_ml_strategy, create_portfolio_strategy
        
        strategies = {
            'technical': [
                {'name': 'Moving Average Crossover', 'id': 'moving_average_crossover'},
                {'name': 'RSI Mean Reversion', 'id': 'rsi_mean_reversion'},
                {'name': 'MACD Strategy', 'id': 'macd'},
                {'name': 'Bollinger Bands', 'id': 'bollinger_bands'}
            ],
            'machine_learning': [
                {'name': 'Random Forest', 'id': 'random_forest'},
                {'name': 'XGBoost', 'id': 'xgboost'},
                {'name': 'SVM', 'id': 'svm'}
            ],
            'portfolio': [
                {'name': 'Basic Portfolio', 'id': 'basic'},
                {'name': 'Risk Parity', 'id': 'risk_parity'},
                {'name': 'Momentum Portfolio', 'id': 'momentum'}
            ]
        }
        
        return jsonify({
            'success': True,
            'strategies': strategies
        })
        
    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/strategies/backtest', methods=['POST'])
@login_required
def backtest_strategy():
    """Run backtest for a strategy"""
    try:
        data = request.get_json()
        strategy_type = data.get('strategy_type')  # technical, ml, portfolio
        strategy_id = data.get('strategy_id')
        parameters = data.get('parameters', {})
        
        # TODO: Load market data based on request
        # For now, return mock results
        
        mock_results = {
            'performance': {
                'total_return': 0.156,
                'annual_return': 0.234,
                'sharpe_ratio': 1.89,
                'max_drawdown': -0.089,
                'win_rate': 0.634,
                'total_trades': 147
            },
            'equity_curve': [
                {'date': '2023-01-01', 'equity': 10000},
                {'date': '2023-06-01', 'equity': 11200},
                {'date': '2023-12-01', 'equity': 11560}
            ],
            'trades': [
                {'entry_date': '2023-01-05', 'exit_date': '2023-01-10', 'pnl': 250},
                {'entry_date': '2023-01-15', 'exit_date': '2023-01-20', 'pnl': -120}
            ]
        }
        
        return jsonify({
            'success': True,
            'results': mock_results
        })
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/optimization/run', methods=['POST'])
@login_required
def run_optimization():
    """Run strategy optimization"""
    try:
        data = request.get_json()
        strategy_id = data.get('strategy_id')
        parameter_space = data.get('parameter_space', {})
        optimization_method = data.get('method', 'genetic')
        
        # TODO: Implement actual optimization
        # For now, return mock results
        
        mock_optimization = {
            'best_parameters': {'fast_window': 12, 'slow_window': 26},
            'best_score': 1.89,
            'optimization_history': [
                {'iteration': 1, 'score': 1.2},
                {'iteration': 10, 'score': 1.5},
                {'iteration': 20, 'score': 1.89}
            ]
        }
        
        return jsonify({
            'success': True,
            'results': mock_optimization
        })
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/portfolio/analyze', methods=['POST'])
@login_required
def analyze_portfolio():
    """Analyze portfolio performance"""
    try:
        data = request.get_json()
        strategies = data.get('strategies', [])
        allocations = data.get('allocations', {})
        
        # TODO: Implement portfolio analysis
        # For now, return mock results
        
        mock_analysis = {
            'portfolio_metrics': {
                'total_return': 0.189,
                'volatility': 0.156,
                'sharpe_ratio': 1.21,
                'max_drawdown': -0.067
            },
            'strategy_contributions': {
                'strategy_1': 0.045,
                'strategy_2': 0.067,
                'strategy_3': 0.077
            },
            'correlation_matrix': {
                'strategy_1': {'strategy_1': 1.0, 'strategy_2': 0.3, 'strategy_3': 0.1},
                'strategy_2': {'strategy_1': 0.3, 'strategy_2': 1.0, 'strategy_3': 0.2},
                'strategy_3': {'strategy_1': 0.1, 'strategy_2': 0.2, 'strategy_3': 1.0}
            }
        }
        
        return jsonify({
            'success': True,
            'analysis': mock_analysis
        })
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/market/data')
@login_required
def get_market_data():
    """Get market data for symbols"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '1h')
        limit = int(request.args.get('limit', 100))
        
        # TODO: Fetch actual market data
        # For now, return mock data
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
        mock_data = []
        
        for i, date in enumerate(dates):
            mock_data.append({
                'date': date.isoformat(),
                'open': 50000 + i * 10,
                'high': 50200 + i * 10,
                'low': 49800 + i * 10,
                'close': 50100 + i * 10,
                'volume': 1000 + i * 5
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'data': mock_data
        })
        
    except Exception as e:
        logger.error(f"Market data fetch failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/system/status')
@login_required
def system_status():
    """Get system status and metrics"""
    try:
        from mousatrade.utils.version import get_version_manager
        
        vm = get_version_manager()
        version_info = vm.get_current_version()
        
        status = {
            'version': version_info['version'],
            'status': 'running',
            'uptime': '24 hours',  # TODO: Calculate actual uptime
            'memory_usage': '45%',
            'cpu_usage': '23%',
            'active_strategies': 3,
            'total_trades': 147,
            'last_update': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Admin API routes
@api.route('/admin/users')
@admin_required
def list_users():
    """Get list of all users (admin only)"""
    try:
        # TODO: Implement user listing from database
        users = [
            {'id': 1, 'username': 'admin', 'email': 'admin@mousatrade.com', 'role': 'admin', 'last_login': '2024-01-15'},
            {'id': 2, 'username': 'user1', 'email': 'user1@example.com', 'role': 'user', 'last_login': '2024-01-14'}
        ]
        
        return jsonify({
            'success': True,
            'users': users
        })
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api.route('/admin/system/info')
@admin_required
def system_info():
    """Get detailed system information (admin only)"""
    try:
        import psutil
        import platform
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
        return jsonify({
            'success': True,
            'system_info': system_info
        })
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

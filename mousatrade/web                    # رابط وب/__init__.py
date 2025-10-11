"""
Web Module for Mousa Trading Bot
Web interface, API routes, and authentication system
"""

__version__ = '1.0.0'
__author__ = 'Mousa Trading Bot Team'
__description__ = 'Web interface and API for Mousa Trading Bot'

from .app import create_app, app
from .auth import auth, login_required, admin_required
from .routes import main, api

__all__ = [
    'create_app',
    'app', 
    'auth',
    'login_required',
    'admin_required',
    'main',
    'api'
]

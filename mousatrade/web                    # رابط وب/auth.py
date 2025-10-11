"""
Authentication system for Mousa Trading Bot web interface
"""

from functools import wraps
from flask import Flask, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from typing import Optional, Dict, Any
import logging
import sqlite3
from pathlib import Path

class AuthManager:
    """
    Manages user authentication and authorization
    """
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.logger = self._setup_logging()
        self.secret_key = 'mousa-trading-secret-key'  # In production, use environment variable
        self.token_expiry = datetime.timedelta(hours=24)
        
        if app:
            self.init_app(app)
    
    def _setup_logging(self):
        """Setup authentication logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def init_app(self, app: Flask):
        """Initialize authentication with Flask app"""
        self.app = app
        self.secret_key = app.config.get('SECRET_KEY', self.secret_key)
        self._init_database()
    
    def _init_database(self):
        """Initialize authentication database"""
        db_path = Path('data/auth.db')
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE,
                expires_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create default admin user if not exists
        cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('admin',))
        if cursor.fetchone()[0] == 0:
            admin_password = generate_password_hash('admin123')
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', ('admin', 'admin@mousatrade.com', admin_password, 'admin'))
            self.logger.info("Default admin user created: admin/admin123")
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> bool:
        """Create a new user"""
        try:
            password_hash = generate_password_hash(password)
            conn = sqlite3.connect('data/auth.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, role))
            
            conn.commit()
            conn.close()
            self.logger.info(f"User created: {username}")
            return True
            
        except sqlite3.IntegrityError:
            self.logger.warning(f"User creation failed: {username} already exists")
            return False
        except Exception as e:
            self.logger.error(f"User creation failed: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info"""
        try:
            conn = sqlite3.connect('data/auth.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password_hash, role, is_active
                FROM users WHERE username = ? AND is_active = TRUE
            ''', (username,))
            
            user_data = cursor.fetchone()
            
            if user_data and check_password_hash(user_data[3], password):
                user = {
                    'id': user_data[0],
                    'username': user_data[1],
                    'email': user_data[2],
                    'role': user_data[4],
                    'is_active': user_data[5]
                }
                
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user['id'],))
                conn.commit()
                
                conn.close()
                self.logger.info(f"User authenticated: {username}")
                return user
            else:
                conn.close()
                return None
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return None
    
    def generate_token(self, user_id: int, username: str, role: str) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'exp': datetime.datetime.utcnow() + self.token_expiry
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            conn = sqlite3.connect('data/auth.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, role, is_active, created_at, last_login
                FROM users WHERE id = ?
            ''', (user_id,))
            
            user_data = cursor.fetchone()
            conn.close()
            
            if user_data:
                return {
                    'id': user_data[0],
                    'username': user_data[1],
                    'email': user_data[2],
                    'role': user_data[3],
                    'is_active': user_data[4],
                    'created_at': user_data[5],
                    'last_login': user_data[6]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get user: {e}")
            return None

# Flask authentication decorators
def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if token and token.startswith('Bearer '):
            token = token[7:]  # Remove 'Bearer ' prefix
            user_data = auth.verify_token(token)
            if user_data:
                request.user = user_data
                return f(*args, **kwargs)
        
        # Check session for web routes
        if 'user_id' in session:
            user_data = auth.get_user_by_id(session['user_id'])
            if user_data:
                request.user = user_data
                return f(*args, **kwargs)
        
        if request.is_json:
            return jsonify({'error': 'Authentication required'}), 401
        else:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('main.login'))
    
    return decorated_function

def admin_required(f):
    """Decorator to require admin role for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # First check if user is logged in
        if not hasattr(request, 'user'):
            token = request.headers.get('Authorization')
            if token and token.startswith('Bearer '):
                token = token[7:]
                user_data = auth.verify_token(token)
                if user_data:
                    request.user = user_data
                else:
                    return jsonify({'error': 'Invalid token'}), 401
            elif 'user_id' in session:
                user_data = auth.get_user_by_id(session['user_id'])
                if user_data:
                    request.user = user_data
                else:
                    session.clear()
                    return redirect(url_for('main.login'))
            else:
                if request.is_json:
                    return jsonify({'error': 'Authentication required'}), 401
                else:
                    flash('Please log in to access this page', 'warning')
                    return redirect(url_for('main.login'))
        
        # Check if user is admin
        if request.user.get('role') != 'admin':
            if request.is_json:
                return jsonify({'error': 'Admin access required'}), 403
            else:
                flash('Admin access required', 'error')
                return redirect(url_for('main.dashboard'))
        
        return f(*args, **kwargs)
    
    return decorated_function

# Global auth instance
auth = AuthManager()

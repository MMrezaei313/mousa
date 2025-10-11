import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from .base_strategy import BaseStrategy

class MLStrategy(BaseStrategy):
    """
    Base Machine Learning Strategy Class
    """
    
    def __init__(self, model=None, lookback_period: int = 50, 
                 feature_lags: List[int] = None, **kwargs):
        params = {
            'lookback_period': lookback_period,
            'feature_lags': feature_lags or [1, 2, 3, 5, 10]
        }
        params.update(kwargs)
        super().__init__("MLStrategy", **params)
        
        self.model = model
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for ML model"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['high_low_ratio'] = data['high'] / data['low']
        features['open_close_ratio'] = data['open'] / data['close']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = data['close'].rolling(period).mean()
            features[f'ma_ratio_{period}'] = data['close'] / features[f'ma_{period}']
            features[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_ma_{period}']
        
        # RSI
        features['rsi'] = self._calculate_rsi(data['close'])
        
        # MACD
        macd_features = self._calculate_macd(data['close'])
        features = pd.concat([features, macd_features], axis=1)
        
        # Bollinger Bands position
        bb_features = self._calculate_bollinger_features(data['close'])
        features = pd.concat([features, bb_features], axis=1)
        
        # Lagged features
        for lag in self.params['feature_lags']:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio_5'].shift(lag)
        
        # Remove NaN values
        features = features.dropna()
        
        self.feature_columns = [col for col in features.columns if col != 'target']
        return features
    
    def create_target(self, data: pd.DataFrame, prediction_horizon: int = 1) -> pd.Series:
        """Create target variable for classification"""
        future_returns = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        target = (future_returns > 0).astype(int)  # 1 if price goes up, 0 if down
        target.name = 'target'
        return target
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate MACD features"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram
        }, index=prices.index)
    
    def _calculate_bollinger_features(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate Bollinger Bands features"""
        middle_band = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        upper_band = middle_band + (std * 2)
        lower_band = middle_band - (std * 2)
        
        return pd.DataFrame({
            'bb_middle': middle_band,
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_position': (prices - lower_band) / (upper_band - lower_band),
            'bb_width': (upper_band - lower_band) / middle_band
        }, index=prices.index)
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the ML model"""
        self.logger.info("Training ML model...")
        
        # Create features and target
        features = self.create_features(data)
        target = self.create_target(data)
        
        # Align indices
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        if len(features) < 100:
            raise ValueError("Insufficient data for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features[self.feature_columns], target, 
            test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': self._get_feature_importance(features.columns),
            'training_samples': len(X_train),
            'testing_samples': len(X_test)
        }
    
    def _get_feature_importance(self, feature_names) -> pd.Series:
        """Get feature importance from model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return pd.Series()
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using ML model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")
        
        # Create features
        features = self.create_features(data)
        
        if features.empty:
            return pd.DataFrame()
        
        # Scale features and make predictions
        features_scaled = self.scaler.transform(features[self.feature_columns])
        predictions = self.model.predict(features_scaled)
        prediction_proba = self.model.predict_proba(features_scaled)
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=features.index)
        signals['prediction'] = predictions
        signals['probability'] = prediction_proba.max(axis=1)
        signals['confidence'] = np.abs(prediction_proba[:, 1] - 0.5) * 2
        
        # Generate positions (1 for buy, -1 for sell, 0 for hold)
        # Use prediction with confidence threshold
        confidence_threshold = 0.3
        signals['position'] = 0
        signals.loc[
            (signals['prediction'] == 1) & (signals['confidence'] > confidence_threshold), 
            'position'
        ] = 1
        signals.loc[
            (signals['prediction'] == 0) & (signals['confidence'] > confidence_threshold), 
            'position'
        ] = -1
        
        # Add feature values for analysis
        for col in self.feature_columns[:5]:  # Add top 5 features
            signals[col] = features[col]
        
        self.signals = signals
        return signals

class RandomForestStrategy(MLStrategy):
    """
    Random Forest based trading strategy
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, **kwargs):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        params.update(kwargs)
        super().__init__(model=model, **params)
        self.name = "RandomForestStrategy"

class XGBoostStrategy(MLStrategy):
    """
    XGBoost based trading strategy
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, **kwargs):
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        except ImportError:
            self.logger.warning("XGBoost not available, using RandomForest")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        params.update(kwargs)
        super().__init__(model=model, **params)
        self.name = "XGBoostStrategy"

class SVMTradingStrategy(MLStrategy):
    """
    Support Vector Machine based trading strategy
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', **kwargs):
        model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        params = {
            'C': C,
            'kernel': kernel
        }
        params.update(kwargs)
        super().__init__(model=model, **params)
        self.name = "SVMTradingStrategy"

# ML Strategy factory
def create_ml_strategy(strategy_type: str = 'random_forest', **params) -> MLStrategy:
    """Factory function for ML strategies"""
    strategies = {
        'random_forest': RandomForestStrategy,
        'xgboost': XGBoostStrategy,
        'svm': SVMTradingStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown ML strategy: {strategy_type}")
    
    return strategies[strategy_type](**params)

# price_prediction_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from torch.utils.data import Dataset, DataLoader
import math
from scipy import stats
import ta

warnings.filterwarnings('ignore')

class PredictionType(Enum):
    PRICE = "price"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    RETURN = "return"

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class PricePrediction:
    predicted_price: float
    prediction_type: PredictionType
    confidence: float
    confidence_level: ConfidenceLevel
    upper_bound: float
    lower_bound: float
    prediction_horizon: int
    timestamp: pd.Timestamp
    features_importance: Dict[str, float]

@dataclass
class MultiHorizonPrediction:
    symbol: str
    predictions: List[PricePrediction]
    trend_direction: str
    trend_strength: float
    key_support: float
    key_resistance: float
    market_regime: str

class MultiScaleAttention(nn.Module):
    """Multi-scale attention for capturing different time frame patterns"""
    
    def __init__(self, d_model: int, n_heads: int, scales: List[int] = None, dropout: float = 0.1):
        super(MultiScaleAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales or [1, 3, 5, 10]
        self.head_dim = d_model // n_heads
        
        # Multi-scale projections
        self.w_queries = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in self.scales
        ])
        self.w_keys = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in self.scales
        ])
        self.w_values = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in self.scales
        ])
        
        self.output_projection = nn.Linear(d_model * len(self.scales), d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        multi_scale_outputs = []
        
        for scale, w_q, w_k, w_v in zip(self.scales, self.w_queries, self.w_keys, self.w_values):
            # Create scaled sequences
            if scale > 1:
                # Downsample for larger scales
                scaled_seq_len = seq_len // scale
                if scaled_seq_len == 0:
                    continue
                    
                scaled_x = x[:, ::scale, :][:, :scaled_seq_len, :]
                scaled_seq_len = scaled_x.shape[1]
            else:
                scaled_x = x
                scaled_seq_len = seq_len
            
            # Project queries, keys, values
            Q = w_q(scaled_x).view(batch_size, scaled_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = w_k(scaled_x).view(batch_size, scaled_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V = w_v(scaled_x).view(batch_size, scaled_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention
            scale_output = torch.matmul(attention_weights, V)
            scale_output = scale_output.transpose(1, 2).contiguous().view(
                batch_size, scaled_seq_len, self.d_model
            )
            
            # Upsample back to original sequence length if needed
            if scale > 1:
                scale_output = torch.nn.functional.interpolate(
                    scale_output.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            multi_scale_outputs.append(scale_output)
        
        # Combine multi-scale outputs
        combined_output = torch.cat(multi_scale_outputs, dim=-1)
        output = self.output_projection(combined_output)
        
        return self.layer_norm(output + x)

class ProbabilisticOutputLayer(nn.Module):
    """Probabilistic output layer for uncertainty estimation"""
    
    def __init__(self, d_model: int, output_dim: int, num_mixtures: int = 3):
        super(ProbabilisticOutputLayer, self).__init__()
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures
        
        # Mixture density network
        self.mixture_weights = nn.Linear(d_model, num_mixtures)
        self.means = nn.Linear(d_model, output_dim * num_mixtures)
        self.std_devs = nn.Linear(d_model, output_dim * num_mixtures)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Predict mixture parameters
        weights = torch.softmax(self.mixture_weights(x), dim=-1)  # (batch, num_mixtures)
        means = self.means(x).view(batch_size, self.num_mixtures, self.output_dim)  # (batch, num_mixtures, output_dim)
        std_devs = torch.exp(self.std_devs(x)).view(batch_size, self.num_mixtures, self.output_dim)  # (batch, num_mixtures, output_dim)
        
        return weights, means, std_devs

class PricePredictionTransformer(nn.Module):
    """
    Advanced Transformer for financial price prediction with uncertainty quantification
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 prediction_horizon: int = 5,
                 use_multiscale: bool = True,
                 use_probabilistic: bool = True):
        super(PricePredictionTransformer, self).__init__()
        
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.use_probabilistic = use_probabilistic
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Multi-scale attention layers
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale_attention = MultiScaleAttention(d_model, n_heads)
        
        # Standard transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        if use_probabilistic:
            self.probabilistic_output = ProbabilisticOutputLayer(d_model, prediction_horizon)
        else:
            self.regression_output = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, prediction_horizon)
            )
        
        # Feature importance projection
        self.feature_importance = nn.Linear(d_model, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, input_dim = x.shape
        
        # Input projection
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Multi-scale attention
        if self.use_multiscale:
            x = self.multiscale_attention(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Get last hidden state for prediction
        last_hidden = encoded[:, -1, :]
        
        # Feature importance
        feature_importance = torch.sigmoid(self.feature_importance(last_hidden))
        
        # Output predictions
        if self.use_probabilistic:
            weights, means, std_devs = self.probabilistic_output(last_hidden)
            output_dict = {
                'mixture_weights': weights,
                'means': means,
                'std_devs': std_devs,
                'feature_importance': feature_importance
            }
        else:
            predictions = self.regression_output(last_hidden)
            output_dict = {
                'predictions': predictions,
                'feature_importance': feature_importance
            }
        
        return output_dict

class FinancialPriceDataset(Dataset):
    """Dataset for financial price prediction"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 features: List[str],
                 target: str,
                 sequence_length: int = 60,
                 prediction_horizon: int = 5,
                 normalize: bool = True):
        
        self.data = data[features + [target]].values
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Normalize data
        if normalize:
            self.data_mean = np.mean(self.data, axis=0)
            self.data_std = np.std(self.data, axis=0)
            self.data = (self.data - self.data_mean) / (self.data_std + 1e-8)
        
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create input-output sequences for multi-horizon prediction"""
        sequences = []
        
        for i in range(len(self.data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence (features only)
            seq_input = self.data[i:i + self.sequence_length, :-1]
            
            # Output (multiple future target values)
            seq_output = self.data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, -1]
            
            sequences.append((seq_input, seq_output))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_input, seq_output = self.sequences[idx]
        return torch.FloatTensor(seq_input), torch.FloatTensor(seq_output)

class PositionalEncoding(nn.Module):
    """Positional encoding for time series"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class AdvancedPricePredictor:
    """
    Advanced price prediction system using transformer architecture
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'sequence_length': 60,
            'prediction_horizon': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'patience': 15,
            'use_multiscale': True,
            'use_probabilistic': True,
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_names = None
        self.data_stats = {}
        self.is_trained = False
    
    def create_advanced_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for price prediction
        
        Args:
            price_data: OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        data = price_data.copy()
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['price_range'] = (data['high'] - data['low']) / data['close']
        data['body_size'] = (data['close'] - data['open']) / data['close']
        
        # Multiple time frame returns
        for period in [1, 3, 5, 10]:
            data[f'returns_{period}'] = data['close'].pct_change(period)
            data[f'volatility_{period}'] = data['returns'].rolling(period).std()
        
        # Moving averages and ratios
        for window in [5, 10, 20, 50]:
            data[f'sma_{window}'] = data['close'].rolling(window).mean()
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
            data[f'price_vs_sma_{window}'] = (data['close'] - data[f'sma_{window}']) / data[f'sma_{window}']
        
        # Technical indicators
        data = self._add_technical_indicators(data)
        
        # Statistical features
        data['rolling_skew_20'] = data['returns'].rolling(20).skew()
        data['rolling_kurt_20'] = data['returns'].rolling(20).kurt()
        data['z_score_20'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_sma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            data['price_volume_corr'] = data['close'].rolling(20).corr(data['volume'])
        
        # Market regime features
        data['trend_strength'] = self._calculate_trend_strength(data)
        data['volatility_regime'] = self._calculate_volatility_regime(data)
        data['market_regime'] = self._classify_market_regime(data)
        
        # Cyclical features
        data['day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag) if 'volume' in data.columns else 0
        
        # Target variable (future returns)
        data['target_returns'] = data['close'].pct_change(self.config['prediction_horizon']).shift(-self.config['prediction_horizon'])
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # RSI
            data['rsi_14'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
            data['rsi_28'] = ta.momentum.RSIIndicator(data['close'], window=28).rsi()
            
            # MACD
            macd = ta.trend.MACD(data['close'])
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'], window=20)
            data['bb_upper'] = bb.bollinger_hband()
            data['bb_lower'] = bb.bollinger_lband()
            data['bb_middle'] = bb.bollinger_mavg()
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # ATR
            if all(col in data.columns for col in ['high', 'low', 'close']):
                atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'])
                data['atr'] = atr.average_true_range()
                data['atr_ratio'] = data['atr'] / data['close']
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
            data['stoch_k'] = stoch.stoch()
            data['stoch_d'] = stoch.stoch_signal()
            
            # ADX
            adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'])
            data['adx'] = adx.adx()
            
        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")
        
        return data
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using multiple methods"""
        if 'sma_20' not in data.columns or 'sma_50' not in data.columns:
            return pd.Series(0.5, index=data.index)
        
        # Price position relative to MAs
        price_vs_sma20 = (data['close'] - data['sma_20']) / data['sma_20']
        price_vs_sma50 = (data['close'] - data['sma_50']) / data['sma_50']
        
        # MA alignment
        ma_alignment = (data['sma_20'] > data['sma_50']).astype(float)
        
        # Combine signals
        trend_strength = (abs(price_vs_sma20) + abs(price_vs_sma50)) / 2 * ma_alignment
        
        return trend_strength.clip(0, 1)
    
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime"""
        if 'volatility_20' not in data.columns:
            return pd.Series(0, index=data.index)
        
        vol_median = data['volatility_20'].median()
        vol_std = data['volatility_20'].std()
        
        # Normalize volatility to 0-1 range
        volatility_score = (data['volatility_20'] - vol_median) / vol_std
        return (np.tanh(volatility_score) + 1) / 2  # Convert to 0-1
    
    def _classify_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """Classify market regime"""
        regime = pd.Series('neutral', index=data.index)
        
        if 'trend_strength' in data.columns and 'volatility_20' in data.columns:
            # Simple regime classification
            high_trend = data['trend_strength'] > 0.7
            low_trend = data['trend_strength'] < 0.3
            high_vol = data['volatility_20'] > data['volatility_20'].quantile(0.7)
            low_vol = data['volatility_20'] < data['volatility_20'].quantile(0.3)
            
            regime[high_trend & low_vol] = 'trending'
            regime[high_trend & high_vol] = 'volatile_trend'
            regime[low_trend & high_vol] = 'volatile_range'
            regime[low_trend & low_vol] = 'ranging'
        
        return regime
    
    def train(self,
              train_data: pd.DataFrame,
              val_data: pd.DataFrame = None,
              target_column: str = 'target_returns',
              feature_columns: List[str] = None) -> Dict[str, List[float]]:
        """
        Train the price prediction transformer
        
        Args:
            train_data: Training data
            val_data: Validation data
            target_column: Target column name
            feature_columns: List of feature columns
            
        Returns:
            Training history
        """
        
        # Prepare features
        if feature_columns is None:
            exclude_cols = [target_column, 'date', 'timestamp', 'datetime', 'market_regime']
            feature_columns = [col for col in train_data.columns 
                             if col not in exclude_cols and train_data[col].dtype in [np.float64, np.int64]]
        
        self.feature_names = feature_columns
        self.data_stats['feature_columns'] = feature_columns
        
        # Create datasets
        train_dataset = FinancialPriceDataset(
            data=train_data,
            features=feature_columns,
            target=target_column,
            sequence_length=self.config['sequence_length'],
            prediction_horizon=self.config['prediction_horizon'],
            normalize=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        val_loader = None
        if val_data is not None:
            val_dataset = FinancialPriceDataset(
                data=val_data,
                features=feature_columns,
                target=target_column,
                sequence_length=self.config['sequence_length'],
                prediction_horizon=self.config['prediction_horizon'],
                normalize=True
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        # Initialize model
        self.model = PricePredictionTransformer(
            input_dim=len(feature_columns),
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            prediction_horizon=self.config['prediction_horizon'],
            use_multiscale=self.config['use_multiscale'],
            use_probabilistic=self.config['use_probabilistic']
        ).to(self.device)
        
        # Loss function and optimizer
        if self.config['use_probabilistic']:
            criterion = self._mixture_density_loss
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if self.config['use_probabilistic']:
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs['predictions'], batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        
                        if self.config['use_probabilistic']:
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs['predictions'], batch_y)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_price_predictor.pth')
                else:
                    patience_counter += 1
                
                scheduler.step(val_loss)
            
            if patience_counter >= self.config['patience']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_price_predictor.pth'))
        self.is_trained = True
        
        return history
    
    def _mixture_density_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Loss function for mixture density network"""
        weights = outputs['mixture_weights']  # (batch, num_mixtures)
        means = outputs['means']  # (batch, num_mixtures, horizon)
        std_devs = outputs['std_devs']  # (batch, num_mixtures, horizon)
        
        batch_size, num_mixtures, horizon = means.shape
        targets = targets.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, horizon)
        targets = targets.expand(-1, num_mixtures, -1, horizon)
        
        # Normal distribution log probability
        log_prob = -0.5 * ((targets - means) / std_devs) ** 2 - torch.log(std_devs) - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(dim=-1)  # Sum over horizon
        
        # Weighted log likelihood
        weighted_log_prob = log_prob + torch.log(weights)
        max_log_prob = weighted_log_prob.max(dim=1, keepdim=True)[0]
        weighted_log_prob = weighted_log_prob - max_log_prob
        
        # Negative log likelihood
        nll = -torch.log(torch.sum(torch.exp(weighted_log_prob), dim=1)) - max_log_prob.squeeze()
        
        return nll.mean()
    
    def predict(self,
                data: pd.DataFrame,
                current_price: float,
                prediction_type: PredictionType = PredictionType.PRICE) -> MultiHorizonPrediction:
        """
        Make multi-horizon price predictions
        
        Args:
            data: Historical data
            current_price: Current price for conversion
            prediction_type: Type of prediction to return
            
        Returns:
            MultiHorizonPrediction object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        
        # Prepare input data
        input_data = data[self.feature_names].tail(self.config['sequence_length']).values
        input_data = (input_data - input_data.mean(axis=0)) / (input_data.std(axis=0) + 1e-8)
        
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Process predictions
        if self.config['use_probabilistic']:
            predictions = self._process_probabilistic_predictions(outputs, current_price, prediction_type)
        else:
            predictions = self._process_deterministic_predictions(outputs, current_price, prediction_type)
        
        # Calculate additional market insights
        trend_direction, trend_strength = self._analyze_trend(predictions)
        support, resistance = self._calculate_support_resistance(data)
        market_regime = self._determine_market_regime(data)
        
        return MultiHorizonPrediction(
            symbol="PREDICTION",
            predictions=predictions,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            key_support=support,
            key_resistance=resistance,
            market_regime=market_regime
        )
    
    def _process_probabilistic_predictions(self,
                                         outputs: Dict[str, torch.Tensor],
                                         current_price: float,
                                         prediction_type: PredictionType) -> List[PricePrediction]:
        """Process probabilistic model outputs"""
        weights = outputs['mixture_weights'].cpu().numpy()[0]  # (num_mixtures)
        means = outputs['means'].cpu().numpy()[0]  # (num_mixtures, horizon)
        std_devs = outputs['std_devs'].cpu().numpy()[0]  # (num_mixtures, horizon)
        feature_importance = outputs['feature_importance'].cpu().numpy()[0]  # (input_dim)
        
        predictions = []
        
        for horizon in range(self.config['prediction_horizon']):
            # Calculate weighted mean and variance for this horizon
            horizon_means = means[:, horizon]
            horizon_stds = std_devs[:, horizon]
            
            # Expected value (weighted mean of mixture means)
            expected_value = np.sum(weights * horizon_means)
            
            # Variance calculation for mixture
            mean_of_mixture = np.sum(weights * horizon_means)
            variance = np.sum(weights * (horizon_stds**2 + horizon_means**2)) - mean_of_mixture**2
            
            # Convert to price if needed
            if prediction_type == PredictionType.PRICE:
                predicted_value = current_price * (1 + expected_value)
                upper_bound = current_price * (1 + expected_value + 2 * np.sqrt(variance))
                lower_bound = current_price * (1 + expected_value - 2 * np.sqrt(variance))
            else:
                predicted_value = expected_value
                upper_bound = expected_value + 2 * np.sqrt(variance)
                lower_bound = expected_value - 2 * np.sqrt(variance)
            
            # Calculate confidence
            confidence = 1 / (1 + np.sqrt(variance))
            confidence_level = self._get_confidence_level(confidence)
            
            # Create feature importance dictionary
            feature_importance_dict = {
                feature: importance for feature, importance in zip(self.feature_names, feature_importance)
            }
            
            prediction = PricePrediction(
                predicted_price=predicted_value,
                prediction_type=prediction_type,
                confidence=confidence,
                confidence_level=confidence_level,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                prediction_horizon=horizon + 1,
                timestamp=pd.Timestamp.now(),
                features_importance=feature_importance_dict
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _process_deterministic_predictions(self,
                                         outputs: Dict[str, torch.Tensor],
                                         current_price: float,
                                         prediction_type: PredictionType) -> List[PricePrediction]:
        """Process deterministic model outputs"""
        predictions_np = outputs['predictions'].cpu().numpy()[0]  # (horizon)
        feature_importance = outputs['feature_importance'].cpu().numpy()[0]  # (input_dim)
        
        predictions = []
        
        for horizon, pred in enumerate(predictions_np):
            # Convert to price if needed
            if prediction_type == PredictionType.PRICE:
                predicted_value = current_price * (1 + pred)
                # Simple bounds (could be improved)
                upper_bound = predicted_value * 1.02
                lower_bound = predicted_value * 0.98
            else:
                predicted_value = pred
                upper_bound = pred + 0.01
                lower_bound = pred - 0.01
            
            # Simple confidence estimation
            confidence = max(0.1, 1 - abs(pred) * 2)
            confidence_level = self._get_confidence_level(confidence)
            
            # Create feature importance dictionary
            feature_importance_dict = {
                feature: importance for feature, importance in zip(self.feature_names, feature_importance)
            }
            
            prediction = PricePrediction(
                predicted_price=predicted_value,
                prediction_type=prediction_type,
                confidence=confidence,
                confidence_level=confidence_level,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                prediction_horizon=horizon + 1,
                timestamp=pd.Timestamp.now(),
                features_importance=feature_importance_dict
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        thresholds = self.config['confidence_thresholds']
        
        if confidence >= thresholds['high']:
            return ConfidenceLevel.HIGH
        elif confidence >= thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _analyze_trend(self, predictions: List[PricePrediction]) -> Tuple[str, float]:
        """Analyze trend from predictions"""
        if len(predictions) < 2:
            return "neutral", 0.5
        
        prices = [p.predicted_price for p in predictions]
        
        # Simple linear trend
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize slope to trend strength
        price_range = max(prices) - min(prices)
        if price_range > 0:
            trend_strength = min(abs(slope) / price_range * len(prices), 1.0)
        else:
            trend_strength = 0.0
        
        if slope > 0:
            direction = "bullish"
        elif slope < 0:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return direction, trend_strength
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate key support and resistance levels"""
        if len(data) < 20:
            return data['close'].min(), data['close'].max()
        
        # Simple support/resistance using recent highs and lows
        recent_high = data['high'].tail(20).max()
        recent_low = data['low'].tail(20).min()
        current_price = data['close'].iloc[-1]
        
        support = recent_low * 0.99
        resistance = recent_high * 1.01
        
        return support, resistance
    
    def _determine_market_regime(self, data: pd.DataFrame) -> str:
        """Determine current market regime"""
        if 'market_regime' in data.columns:
            return data['market_regime'].iloc[-1]
        
        # Simple regime detection
        volatility = data['close'].pct_change().std()
        trend_strength = abs(data['close'].pct_change(5).iloc[-1])
        
        if volatility > data['close'].pct_change().std() * 1.5:
            return "high_volatility"
        elif trend_strength > 0.05:
            return "trending"
        else:
            return "ranging"

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    n_points = len(dates)
    
    # Synthetic price data with trend and seasonality
    trend = np.linspace(100, 200, n_points)
    noise = np.random.normal(0, 2, n_points)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_points) / 30)
    
    price = trend + seasonality + noise
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': price * (1 + np.random.normal(0, 0.01, n_points)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.015, n_points))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.015, n_points))),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, n_points)
    })
    sample_data.set_index('date', inplace=True)
    
    # Initialize and train model
    predictor = AdvancedPricePredictor()
    
    # Create features
    feature_data = predictor.create_advanced_features(sample_data)
    
    # Split data
    train_size = int(0.8 * len(feature_data))
    train_data = feature_data[:train_size]
    test_data = feature_data[train_size:]
    
    print("Training price prediction model...")
    history = predictor.train(
        train_data=train_data,
        val_data=test_data,
        target_column='target_returns'
    )
    
    # Make predictions
    current_price = sample_data['close'].iloc[-1]
    predictions = predictor.predict(test_data, current_price, PredictionType.PRICE)
    
    print(f"\nPrice Predictions for {predictions.symbol}:")
    print(f"Current Price: {current_price:.2f}")
    print(f"Market Regime: {predictions.market_regime}")
    print(f"Trend: {predictions.trend_direction} (Strength: {predictions.trend_strength:.2f})")
    print(f"Support: {predictions.key_support:.2f}, Resistance: {predictions.key_resistance:.2f}")
    
    print("\nMulti-horizon Predictions:")
    for i, pred in enumerate(predictions.predictions):
        print(f"Horizon {i+1}: {pred.predicted_price:.2f} "
              f"[{pred.lower_bound:.2f}, {pred.upper_bound:.2f}] "
              f"({pred.confidence_level.value} confidence)")

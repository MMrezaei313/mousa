# time_series_transformer.py
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

warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """Positional encoding for time series transformer"""
    
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

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model specialized for financial time series forecasting
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 output_dim: int = 1,
                 prediction_horizon: int = 1,
                 use_positional_encoding: bool = True):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, output_dim)
            ) for _ in range(prediction_horizon)
        ])
        
        # Attention weights for interpretability
        self.attention_weights = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask for sequence
            
        Returns:
            Output tensor of shape (batch_size, prediction_horizon, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoding with attention capture
        def hook_fn(module, input, output):
            self.attention_weights = output[1]  # Attention weights
        
        # Register hook to capture attention
        handle = self.transformer_encoder.layers[-1].self_attn.register_forward_hook(hook_fn)
        
        # Apply transformer
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Remove hook
        handle.remove()
        
        # Multi-horizon prediction
        outputs = []
        for i in range(self.prediction_horizon):
            # Use last hidden state for prediction
            last_hidden = encoded[:, -1, :]
            output = self.output_layers[i](last_hidden)
            outputs.append(output.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

class FinancialTimeSeriesDataset(Dataset):
    """Dataset for financial time series"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 features: List[str],
                 target: str,
                 sequence_length: int = 60,
                 prediction_horizon: int = 1,
                 normalize: bool = True):
        
        self.data = data[features + [target]].values
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        
        # Normalize data
        if normalize:
            self.data_mean = np.mean(self.data, axis=0)
            self.data_std = np.std(self.data, axis=0)
            self.data = (self.data - self.data_mean) / (self.data_std + 1e-8)
        
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create input-output sequences"""
        sequences = []
        
        for i in range(len(self.data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq_input = self.data[i:i + self.sequence_length, :-1]  # Exclude target
            
            # Output (multiple future steps)
            seq_output = self.data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, -1]
            
            sequences.append((seq_input, seq_output))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_input, seq_output = self.sequences[idx]
        return torch.FloatTensor(seq_input), torch.FloatTensor(seq_output)

class MultiHeadTimeSeriesTransformer:
    """
    Advanced transformer for financial time series with multiple prediction heads
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
            'patience': 10
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
    
    def prepare_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive features for time series prediction
        
        Args:
            price_data: OHLC price data
            
        Returns:
            DataFrame with engineered features
        """
        data = price_data.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['price_change'] = data['close'].diff()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'sma_{window}'] = data['close'].rolling(window).mean()
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
        
        # Rolling statistics
        data['rolling_std_20'] = data['returns'].rolling(20).std()
        data['rolling_skew_20'] = data['returns'].rolling(20).skew()
        data['rolling_kurt_20'] = data['returns'].rolling(20).kurt()
        
        # Volatility features
        data['volatility_5'] = data['returns'].rolling(5).std()
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['volatility_ratio'] = data['volatility_5'] / data['volatility_20']
        
        # Technical indicators
        data = self._add_technical_indicators(data)
        
        # Price position features
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag) if 'volume' in data.columns else 0
        
        # Rolling correlations (if multiple assets)
        if 'volume' in data.columns:
            data['price_volume_corr'] = data['close'].rolling(20).corr(data['volume'])
        
        # Market regime features
        data['trend_strength'] = self._calculate_trend_strength(data)
        data['volatility_regime'] = self._calculate_volatility_regime(data)
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataset"""
        try:
            import ta
            
            # RSI
            data['rsi_14'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(data['close'])
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'])
            data['bb_upper'] = bb.bollinger_hband()
            data['bb_lower'] = bb.bollinger_lband()
            data['bb_middle'] = bb.bollinger_mavg()
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # ATR
            if all(col in data.columns for col in ['high', 'low', 'close']):
                atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'])
                data['atr'] = atr.average_true_range()
            
        except ImportError:
            self.logger.warning("TA-Lib not available, using simplified indicators")
            # Simplified RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi_14'] = 100 - (100 / (1 + rs))
        
        return data
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using multiple moving averages"""
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            price_vs_sma20 = (data['close'] - data['sma_20']) / data['sma_20']
            price_vs_sma50 = (data['close'] - data['sma_50']) / data['sma_50']
            ma_alignment = (data['sma_20'] > data['sma_50']).astype(int)
            trend_strength = (abs(price_vs_sma20) + abs(price_vs_sma50)) / 2 * ma_alignment
            return trend_strength
        return pd.Series(0.5, index=data.index)
    
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime indicator"""
        if 'volatility_20' in data.columns:
            vol_median = data['volatility_20'].median()
            return (data['volatility_20'] > vol_median).astype(int)
        return pd.Series(0, index=data.index)
    
    def train(self, 
              train_data: pd.DataFrame,
              val_data: pd.DataFrame = None,
              target_column: str = 'returns',
              feature_columns: List[str] = None) -> Dict[str, List[float]]:
        """
        Train the transformer model
        
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
            # Exclude target and date columns
            exclude_cols = [target_column, 'date', 'timestamp', 'datetime']
            feature_columns = [col for col in train_data.columns 
                             if col not in exclude_cols and train_data[col].dtype in [np.float64, np.int64]]
        
        self.feature_names = feature_columns
        
        # Create datasets
        train_dataset = FinancialTimeSeriesDataset(
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
            val_dataset = FinancialTimeSeriesDataset(
                data=val_data,
                features=feature_columns,
                target=target_column,
                sequence_length=self.config['sequence_length'],
                prediction_horizon=self.config['prediction_horizon'],
                normalize=True
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        # Initialize model
        self.model = TimeSeriesTransformer(
            input_dim=len(feature_columns),
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            output_dim=1,
            prediction_horizon=self.config['prediction_horizon']
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
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
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_transformer_model.pth')
                else:
                    patience_counter += 1
                
                scheduler.step(val_loss)
            
            if patience_counter >= self.config['patience']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_transformer_model.pth'))
        self.is_trained = True
        
        return history
    
    def predict(self, 
                data: pd.DataFrame,
                sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            data: Input data
            sequence_length: Sequence length for prediction
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if sequence_length is None:
            sequence_length = self.config['sequence_length']
        
        self.model.eval()
        
        # Prepare input data
        input_data = data[self.feature_names].values
        input_data = (input_data - input_data.mean(axis=0)) / (input_data.std(axis=0) + 1e-8)
        
        # Create sequences for prediction
        sequences = []
        for i in range(len(input_data) - sequence_length + 1):
            seq = input_data[i:i + sequence_length]
            sequences.append(seq)
        
        if not sequences:
            raise ValueError("Insufficient data for prediction")
        
        sequences = torch.FloatTensor(np.array(sequences)).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(sequences)
            predictions = predictions.cpu().numpy()
        
        # Calculate confidence scores (using prediction variance)
        confidence_scores = 1 / (1 + np.var(predictions, axis=1))
        
        return predictions, confidence_scores
    
    def get_attention_weights(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get attention weights for model interpretability
        
        Args:
            data: Input data
            
        Returns:
            Attention weights
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Prepare single sequence for attention analysis
        input_data = data[self.feature_names].tail(self.config['sequence_length']).values
        input_data = (input_data - input_data.mean(axis=0)) / (input_data.std(axis=0) + 1e-8)
        
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if self.model.attention_weights is not None:
            return self.model.attention_weights.cpu().numpy()
        else:
            return np.zeros((1, self.config['nhead'], self.config['sequence_length'], self.config['sequence_length']))
    
    def forecast(self,
                 price_data: pd.DataFrame,
                 steps: int = 10,
                 confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Generate multi-step forecasts with confidence intervals
        
        Args:
            price_data: Historical price data
            steps: Number of steps to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        
        # Prepare features
        feature_data = self.prepare_features(price_data)
        
        # Get the most recent sequence
        recent_data = feature_data.tail(self.config['sequence_length'])
        
        forecasts = []
        lower_bounds = []
        upper_bounds = []
        
        current_sequence = recent_data.copy()
        
        for step in range(steps):
            # Make prediction for next step
            pred, confidence = self.predict(current_sequence)
            next_pred = pred[0, 0]  # First horizon prediction
            
            # Calculate confidence interval
            pred_std = np.sqrt(1 / confidence[0] - 1) if confidence[0] > 0 else 0.1
            z_score = abs(np.percentile(np.random.randn(10000), confidence_level * 100))
            
            margin = z_score * pred_std
            
            forecasts.append(next_pred)
            lower_bounds.append(next_pred - margin)
            upper_bounds.append(next_pred + margin)
            
            # Update sequence for next prediction (using predicted value)
            # This is a simplified approach - in practice, you might want to use
            # more sophisticated methods like MCMC or ensemble forecasting
            
            # Create new row with prediction
            new_row = current_sequence.iloc[-1:].copy()
            # Update the target column with prediction (this depends on your target)
            # new_row[target_column] = next_pred
            # current_sequence = pd.concat([current_sequence.iloc[1:], new_row])
        
        return {
            'forecasts': np.array(forecasts),
            'lower_bounds': np.array(lower_bounds),
            'upper_bounds': np.array(upper_bounds),
            'confidence_scores': np.array(confidence)
        }
    
    def evaluate_model(self, 
                      test_data: pd.DataFrame,
                      target_column: str = 'returns') -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            test_data: Test data
            target_column: Target column name
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        # Prepare test data
        test_features = self.prepare_features(test_data)
        
        # Make predictions
        predictions, confidence = self.predict(test_features)
        
        # Actual values (for the first horizon)
        actual = test_features[target_column].values[self.config['sequence_length']:self.config['sequence_length'] + len(predictions)]
        
        # Calculate metrics
        mse = np.mean((predictions[:, 0] - actual) ** 2)
        mae = np.mean(np.abs(predictions[:, 0] - actual))
        rmse = np.sqrt(mse)
        
        # Direction accuracy
        actual_direction = np.sign(actual)
        pred_direction = np.sign(predictions[:, 0])
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'mean_confidence': np.mean(confidence)
        }

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    n_points = len(dates)
    
    # Synthetic price data
    returns = np.random.normal(0.001, 0.02, n_points)
    price = 100 * np.cumprod(1 + returns)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': price * (1 + np.random.normal(0, 0.01, n_points)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.015, n_points))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.015, n_points))),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, n_points)
    })
    
    # Initialize and train model
    transformer = MultiHeadTimeSeriesTransformer()
    
    # Prepare features
    feature_data = transformer.prepare_features(sample_data)
    
    # Split data
    train_size = int(0.8 * len(feature_data))
    train_data = feature_data[:train_size]
    test_data = feature_data[train_size:]
    
    # Train model
    print("Training transformer model...")
    history = transformer.train(
        train_data=train_data,
        val_data=test_data,
        target_column='returns'
    )
    
    # Evaluate model
    metrics = transformer.evaluate_model(test_data)
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    forecasts = transformer.forecast(sample_data.tail(100), steps=5)
    
    print("Forecasts:", forecasts['forecasts'])
    print("Confidence Intervals:", list(zip(forecasts['lower_bounds'], forecasts['upper_bounds'])))
    
    # Get attention weights for interpretability
    attention_weights = transformer.get_attention_weights(sample_data.tail(100))
    print(f"Attention weights shape: {attention_weights.shape}")

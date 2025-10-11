import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class CorrelationTracker:
    """
    Track correlations between different assets and markets
    """
    
    def __init__(self, lookback_period: int = 30):
        self.lookback_period = lookback_period
        self.correlation_matrix = pd.DataFrame()
        self.price_data = {}
    
    def update_prices(self, symbol: str, prices: pd.Series):
        """Update price data for a symbol"""
        self.price_data[symbol] = prices.tail(self.lookback_period)
    
    def calculate_correlation_matrix(self, symbols: List[str] = None) -> pd.DataFrame:
        """Calculate correlation matrix for given symbols"""
        if symbols is None:
            symbols = list(self.price_data.keys())
        
        # Filter symbols with sufficient data
        valid_symbols = []
        returns_data = {}
        
        for symbol in symbols:
            if symbol in self.price_data and len(self.price_data[symbol]) >= self.lookback_period:
                returns = self.price_data[symbol].pct_change().dropna()
                if len(returns) >= 10:  # Minimum data points
                    returns_data[symbol] = returns
                    valid_symbols.append(symbol)
        
        if len(valid_symbols) < 2:
            return pd.DataFrame()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        return self.correlation_matrix
    
    def find_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find pairs with correlation above threshold"""
        if self.correlation_matrix.empty:
            return []
        
        correlated_pairs = []
        symbols = self.correlation_matrix.columns
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = self.correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    correlated_pairs.append((symbols[i], symbols[j], corr))
        
        return sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def find_uncorrelated_assets(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """Find assets with low correlation for diversification"""
        if self.correlation_matrix.empty:
            return []
        
        uncorrelated_pairs = []
        symbols = self.correlation_matrix.columns
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = abs(self.correlation_matrix.iloc[i, j])
                if corr <= threshold:
                    uncorrelated_pairs.append((symbols[i], symbols[j], corr))
        
        return sorted(uncorrelated_pairs, key=lambda x: x[2])
    
    def calculate_rolling_correlation(self, symbol1: str, symbol2: str, 
                                   window: int = 20) -> pd.Series:
        """Calculate rolling correlation between two symbols"""
        if symbol1 not in self.price_data or symbol2 not in self.price_data:
            return pd.Series()
        
        prices1 = self.price_data[symbol1]
        prices2 = self.price_data[symbol2]
        
        # Align the data
        aligned_data = pd.DataFrame({
            'price1': prices1,
            'price2': prices2
        }).dropna()
        
        if len(aligned_data) < window:
            return pd.Series()
        
        returns1 = aligned_data['price1'].pct_change()
        returns2 = aligned_data['price2'].pct_change()
        
        rolling_corr = returns1.rolling(window=window).corr(returns2)
        return rolling_corr
    
    def detect_correlation_breakdown(self, symbol1: str, symbol2: str, 
                                   threshold: float = 0.5) -> bool:
        """Detect if correlation between two assets has broken down"""
        current_corr = self.correlation_matrix.get(symbol1, {}).get(symbol2, 0)
        return abs(current_corr) < threshold
    
    def get_portfolio_correlation_analysis(self, portfolio: Dict[str, float]) -> Dict:
        """Analyze correlation structure of a portfolio"""
        symbols = list(portfolio.keys())
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            return {}
        
        # Calculate weighted average correlation
        total_weight = sum(portfolio.values())
        weighted_corr = 0
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i != j:
                    weight = portfolio[sym1] * portfolio[sym2]
                    corr = corr_matrix.iloc[i, j]
                    weighted_corr += weight * corr
        
        weighted_corr /= (total_weight ** 2 - sum(w**2 for w in portfolio.values()))
        
        return {
            'weighted_correlation': weighted_corr,
            'correlation_matrix': corr_matrix,
            'highly_correlated_pairs': self.find_highly_correlated_pairs(0.7),
            'diversification_score': 1 - abs(weighted_corr)
        }

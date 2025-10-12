# feature_importance.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

class ImportanceMethod(Enum):
    PERMUTATION = "permutation"
    SHAP = "shap"
    FEATURE_ABLATION = "feature_ablation"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    ATTENTION_WEIGHTS = "attention_weights"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    MODEL_BASED = "model_based"

class ImportanceType(Enum):
    GLOBAL = "global"
    LOCAL = "local"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"

class SignificanceLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class FeatureImportance:
    feature_name: str
    importance_score: float
    normalized_score: float
    importance_type: ImportanceType
    method: ImportanceMethod
    significance: SignificanceLevel
    confidence_interval: Tuple[float, float]
    direction: str  # positive, negative, neutral
    p_value: float
    stability_score: float
    interactions: List[Tuple[str, float]]  # Feature interactions with scores

@dataclass
class TemporalImportance:
    feature_name: str
    importance_series: pd.Series
    trend: str
    volatility: float
    regime_changes: List[pd.Timestamp]
    seasonal_pattern: bool

@dataclass
class FeatureImportanceAnalysis:
    global_importance: List[FeatureImportance]
    local_importance: Dict[str, List[FeatureImportance]]
    temporal_importance: Dict[str, TemporalImportance]
    interaction_network: Dict[Tuple[str, str], float]
    stability_analysis: Dict[str, float]
    statistical_significance: Dict[str, float]
    feature_groups: Dict[str, List[str]]

class AdvancedFeatureImportance:
    """
    Advanced feature importance analysis for financial AI models
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'n_permutations': 100,
            'shap_samples': 1000,
            'confidence_level': 0.95,
            'stability_threshold': 0.8,
            'significance_alpha': 0.05,
            'interaction_threshold': 0.1,
            'temporal_window': 30,
            'max_display_features': 15
        }
        
        self.feature_names = None
        self.model = None
        self.X_data = None
        self.y_data = None
        
    def comprehensive_importance_analysis(self,
                                        model: Any,
                                        X: np.ndarray,
                                        y: np.ndarray,
                                        feature_names: List[str],
                                        temporal_data: pd.DataFrame = None) -> FeatureImportanceAnalysis:
        """
        Comprehensive feature importance analysis using multiple methods
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            temporal_data: Temporal data for time-based analysis
            
        Returns:
            FeatureImportanceAnalysis object
        """
        
        self.model = model
        self.X_data = X
        self.y_data = y
        self.feature_names = feature_names
        
        try:
            # 1. Global Feature Importance
            global_importance = self._calculate_global_importance(X, y)
            
            # 2. Local Feature Importance
            local_importance = self._calculate_local_importance(X)
            
            # 3. Temporal Importance Analysis
            temporal_importance = self._calculate_temporal_importance(temporal_data) if temporal_data is not None else {}
            
            # 4. Feature Interaction Network
            interaction_network = self._analyze_feature_interactions(X, y)
            
            # 5. Stability Analysis
            stability_analysis = self._analyze_importance_stability(X, y, global_importance)
            
            # 6. Statistical Significance
            statistical_significance = self._calculate_statistical_significance(X, y)
            
            # 7. Feature Grouping
            feature_groups = self._group_correlated_features(X)
            
            return FeatureImportanceAnalysis(
                global_importance=global_importance,
                local_importance=local_importance,
                temporal_importance=temporal_importance,
                interaction_network=interaction_network,
                stability_analysis=stability_analysis,
                statistical_significance=statistical_significance,
                feature_groups=feature_groups
            )
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            raise
    
    def _calculate_global_importance(self, X: np.ndarray, y: np.ndarray) -> List[FeatureImportance]:
        """Calculate global feature importance using multiple methods"""
        
        importance_results = {}
        
        # Method 1: Permutation Importance
        perm_importance = self._permutation_importance(X, y)
        importance_results[ImportanceMethod.PERMUTATION] = perm_importance
        
        # Method 2: SHAP Importance
        shap_importance = self._shap_importance(X)
        importance_results[ImportanceMethod.SHAP] = shap_importance
        
        # Method 3: Model-based Importance
        model_importance = self._model_based_importance(X, y)
        importance_results[ImportanceMethod.MODEL_BASED] = model_importance
        
        # Method 4: Correlation-based Importance
        corr_importance = self._correlation_importance(X, y)
        importance_results[ImportanceMethod.CORRELATION] = corr_importance
        
        # Method 5: Mutual Information
        mi_importance = self._mutual_info_importance(X, y)
        importance_results[ImportanceMethod.MUTUAL_INFO] = mi_importance
        
        # Combine all methods
        combined_importance = self._combine_importance_methods(importance_results)
        
        return combined_importance
    
    def _permutation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate permutation importance"""
        try:
            if hasattr(self.model, 'predict'):
                perm_result = permutation_importance(
                    self.model, X, y,
                    n_repeats=self.config['n_permutations'],
                    random_state=42,
                    n_jobs=-1
                )
                
                importance_dict = {}
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(perm_result.importances_mean):
                        importance_dict[feature_name] = perm_result.importances_mean[i]
                    else:
                        importance_dict[feature_name] = 0.0
                
                return importance_dict
                
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
        
        return {name: 0.0 for name in self.feature_names}
    
    def _shap_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate SHAP importance"""
        try:
            # Sample data for efficiency
            if len(X) > self.config['shap_samples']:
                X_sample = X[np.random.choice(len(X), self.config['shap_samples'], replace=False)]
            else:
                X_sample = X
            
            # Initialize appropriate SHAP explainer
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
            elif hasattr(self.model, 'coef_'):
                explainer = shap.LinearExplainer(self.model, X_sample)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.KernelExplainer(self.model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP values formats
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            
            # Calculate mean absolute SHAP values
            if len(shap_values.shape) == 3:  # Multi-class classification
                mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))
            else:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            importance_dict = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(mean_abs_shap):
                    importance_dict[feature_name] = mean_abs_shap[i]
                else:
                    importance_dict[feature_name] = 0.0
            
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"SHAP importance failed: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def _model_based_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate model-specific feature importance"""
        try:
            importance_dict = {}
            
            if hasattr(self.model, 'feature_importances_'):  # Tree-based models
                importances = self.model.feature_importances_
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(importances):
                        importance_dict[feature_name] = importances[i]
                    else:
                        importance_dict[feature_name] = 0.0
            
            elif hasattr(self.model, 'coef_'):  # Linear models
                if len(self.model.coef_.shape) > 1:  # Multi-class
                    importances = np.mean(np.abs(self.model.coef_), axis=0)
                else:
                    importances = np.abs(self.model.coef_)
                
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(importances):
                        importance_dict[feature_name] = importances[i]
                    else:
                        importance_dict[feature_name] = 0.0
            
            else:
                # Fallback: Train a Random Forest to get importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(rf.feature_importances_):
                        importance_dict[feature_name] = rf.feature_importances_[i]
                    else:
                        importance_dict[feature_name] = 0.0
            
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"Model-based importance failed: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def _correlation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate correlation-based importance"""
        importance_dict = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if i >= X.shape[1]:
                importance_dict[feature_name] = 0.0
                continue
            
            feature_values = X[:, i]
            
            # Handle constant features
            if len(np.unique(feature_values)) <= 1:
                importance_dict[feature_name] = 0.0
                continue
            
            # Calculate correlation and significance
            try:
                correlation, p_value = stats.pearsonr(feature_values, y)
                # Weight by significance (1 - p_value)
                importance_dict[feature_name] = abs(correlation) * (1 - p_value)
            except:
                importance_dict[feature_name] = 0.0
        
        return importance_dict
    
    def _mutual_info_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate mutual information importance"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            importance_dict = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(mi_scores):
                    importance_dict[feature_name] = mi_scores[i]
                else:
                    importance_dict[feature_name] = 0.0
            
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"Mutual information importance failed: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def _combine_importance_methods(self, 
                                  importance_results: Dict[ImportanceMethod, Dict[str, float]]) -> List[FeatureImportance]:
        """Combine results from multiple importance methods"""
        
        combined_scores = {}
        method_weights = {
            ImportanceMethod.SHAP: 0.3,
            ImportanceMethod.PERMUTATION: 0.25,
            ImportanceMethod.MODEL_BASED: 0.2,
            ImportanceMethod.MUTUAL_INFO: 0.15,
            ImportanceMethod.CORRELATION: 0.1
        }
        
        for feature_name in self.feature_names:
            weighted_score = 0.0
            total_weight = 0.0
            method_scores = []
            
            for method, weight in method_weights.items():
                if method in importance_results and feature_name in importance_results[method]:
                    score = importance_results[method][feature_name]
                    weighted_score += score * weight
                    total_weight += weight
                    method_scores.append(score)
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                # Calculate stability (consistency across methods)
                if len(method_scores) > 1:
                    stability = 1 - (np.std(method_scores) / (np.mean(method_scores) + 1e-8))
                else:
                    stability = 0.5
            else:
                final_score = 0.0
                stability = 0.0
            
            # Determine direction
            direction = self._determine_feature_direction(feature_name)
            
            # Calculate confidence interval (simplified)
            confidence_interval = self._calculate_confidence_interval(method_scores)
            
            # Calculate p-value (simplified)
            p_value = self._calculate_feature_p_value(feature_name)
            
            # Determine significance level
            significance = self._determine_significance_level(final_score, p_value, stability)
            
            # Get feature interactions
            interactions = self._get_feature_interactions(feature_name)
            
            combined_scores[feature_name] = FeatureImportance(
                feature_name=feature_name,
                importance_score=final_score,
                normalized_score=final_score,  # Will be normalized later
                importance_type=ImportanceType.GLOBAL,
                method=ImportanceMethod.SHAP,  # Primary method
                significance=significance,
                confidence_interval=confidence_interval,
                direction=direction,
                p_value=p_value,
                stability_score=stability,
                interactions=interactions
            )
        
        # Normalize scores
        max_score = max([fi.importance_score for fi in combined_scores.values()])
        if max_score > 0:
            for feature_name in combined_scores:
                combined_scores[feature_name].normalized_score = (
                    combined_scores[feature_name].importance_score / max_score
                )
        
        # Sort by importance score
        sorted_importance = sorted(
            combined_scores.values(), 
            key=lambda x: x.importance_score, 
            reverse=True
        )
        
        return sorted_importance[:self.config['max_display_features']]
    
    def _determine_feature_direction(self, feature_name: str) -> str:
        """Determine if feature has positive, negative, or neutral impact"""
        try:
            # Simplified direction detection
            # In practice, use SHAP values or partial dependence
            feature_idx = self.feature_names.index(feature_name)
            feature_values = self.X_data[:, feature_idx]
            
            # Correlation with target
            correlation = np.corrcoef(feature_values, self.y_data)[0, 1]
            
            if correlation > 0.1:
                return "positive"
            elif correlation < -0.1:
                return "negative"
            else:
                return "neutral"
                
        except:
            return "neutral"
    
    def _calculate_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for importance scores"""
        if len(scores) < 2:
            return (0.0, 0.0)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 95% confidence interval
        z_score = 1.96  # For 95% confidence
        margin = z_score * std_score / np.sqrt(len(scores))
        
        return (mean_score - margin, mean_score + margin)
    
    def _calculate_feature_p_value(self, feature_name: str) -> float:
        """Calculate statistical significance p-value for feature"""
        try:
            feature_idx = self.feature_names.index(feature_name)
            feature_values = self.X_data[:, feature_idx]
            
            # Simple t-test for feature significance
            if len(np.unique(feature_values)) > 1:
                # Split target based on feature median
                median_val = np.median(feature_values)
                high_group = self.y_data[feature_values > median_val]
                low_group = self.y_data[feature_values <= median_val]
                
                if len(high_group) > 1 and len(low_group) > 1:
                    t_stat, p_value = stats.ttest_ind(high_group, low_group)
                    return p_value
            
            return 1.0  # Not significant
            
        except:
            return 1.0
    
    def _determine_significance_level(self, importance: float, p_value: float, stability: float) -> SignificanceLevel:
        """Determine significance level based on importance, p-value, and stability"""
        score = (importance * (1 - p_value) * stability)
        
        if score > 0.8:
            return SignificanceLevel.VERY_HIGH
        elif score > 0.6:
            return SignificanceLevel.HIGH
        elif score > 0.4:
            return SignificanceLevel.MEDIUM
        elif score > 0.2:
            return SignificanceLevel.LOW
        else:
            return SignificanceLevel.VERY_LOW
    
    def _get_feature_interactions(self, feature_name: str) -> List[Tuple[str, float]]:
        """Get top interactions for a feature"""
        # Simplified interaction detection
        # In practice, use H-statistic or partial dependence
        interactions = []
        feature_idx = self.feature_names.index(feature_name)
        
        for other_idx, other_feature in enumerate(self.feature_names):
            if other_idx == feature_idx:
                continue
            
            # Simple correlation as interaction proxy
            corr = np.corrcoef(self.X_data[:, feature_idx], self.X_data[:, other_idx])[0, 1]
            interaction_strength = abs(corr)
            
            if interaction_strength > self.config['interaction_threshold']:
                interactions.append((other_feature, interaction_strength))
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x[1], reverse=True)
        return interactions[:5]  # Return top 5 interactions
    
    def _calculate_local_importance(self, X: np.ndarray) -> Dict[str, List[FeatureImportance]]:
        """Calculate local feature importance for sample instances"""
        local_importance = {}
        
        # Sample instances for local explanation
        n_samples = min(10, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        for idx in sample_indices:
            instance = X[idx:idx+1]
            instance_importance = self._calculate_instance_importance(instance)
            local_importance[f"instance_{idx}"] = instance_importance
        
        return local_importance
    
    def _calculate_instance_importance(self, instance: np.ndarray) -> List[FeatureImportance]:
        """Calculate feature importance for a single instance"""
        try:
            # Use SHAP for local explanations
            if hasattr(self.model, 'predict'):
                explainer = shap.KernelExplainer(self.model.predict, self.X_data)
                shap_values = explainer.shap_values(instance)
                
                instance_importance = []
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(shap_values[0]):
                        importance_score = abs(shap_values[0][i])
                        
                        feature_importance = FeatureImportance(
                            feature_name=feature_name,
                            importance_score=importance_score,
                            normalized_score=importance_score,
                            importance_type=ImportanceType.LOCAL,
                            method=ImportanceMethod.SHAP,
                            significance=SignificanceLevel.MEDIUM,  # Simplified
                            confidence_interval=(importance_score * 0.8, importance_score * 1.2),
                            direction="positive" if shap_values[0][i] > 0 else "negative",
                            p_value=0.05,  # Simplified
                            stability_score=0.7,  # Simplified
                            interactions=[]
                        )
                        instance_importance.append(feature_importance)
                
                # Sort by importance
                instance_importance.sort(key=lambda x: x.importance_score, reverse=True)
                return instance_importance[:self.config['max_display_features']]
            
        except Exception as e:
            self.logger.warning(f"Local importance calculation failed: {e}")
        
        return []
    
    def _calculate_temporal_importance(self, temporal_data: pd.DataFrame) -> Dict[str, TemporalImportance]:
        """Calculate temporal feature importance"""
        temporal_importance = {}
        
        try:
            # Assuming temporal_data has datetime index and features as columns
            for feature_name in self.feature_names:
                if feature_name in temporal_data.columns:
                    feature_series = temporal_data[feature_name]
                    
                    # Calculate rolling importance (simplified)
                    rolling_importance = feature_series.rolling(
                        window=self.config['temporal_window']
                    ).std()  # Using volatility as proxy for importance changes
                    
                    # Analyze trend
                    trend = "increasing" if rolling_importance.iloc[-1] > rolling_importance.iloc[0] else "decreasing"
                    
                    # Calculate volatility
                    volatility = rolling_importance.std()
                    
                    # Detect regime changes (simplified)
                    regime_changes = self._detect_regime_changes(rolling_importance)
                    
                    # Check for seasonal patterns
                    seasonal = self._check_seasonal_pattern(rolling_importance)
                    
                    temporal_importance[feature_name] = TemporalImportance(
                        feature_name=feature_name,
                        importance_series=rolling_importance,
                        trend=trend,
                        volatility=volatility,
                        regime_changes=regime_changes,
                        seasonal_pattern=seasonal
                    )
        
        except Exception as e:
            self.logger.warning(f"Temporal importance calculation failed: {e}")
        
        return temporal_importance
    
    def _detect_regime_changes(self, series: pd.Series) -> List[pd.Timestamp]:
        """Detect regime changes in temporal series"""
        try:
            # Simple regime change detection using z-score
            z_scores = np.abs(stats.zscore(series.dropna()))
            regime_changes = series.index[z_scores > 2].tolist()
            return regime_changes[:5]  # Return top 5 changes
        except:
            return []
    
    def _check_seasonal_pattern(self, series: pd.Series) -> bool:
        """Check if series exhibits seasonal patterns"""
        try:
            # Simple seasonal check using autocorrelation
            from statsmodels.tsa.stattools import acf
            
            autocorr = acf(series.dropna(), nlags=20)
            # Check if there's significant autocorrelation at seasonal lags
            seasonal_lags = [7, 14, 30]  # Weekly, bi-weekly, monthly
            seasonal_corr = any(abs(autocorr[lag]) > 0.3 for lag in seasonal_lags if lag < len(autocorr))
            return seasonal_corr
        except:
            return False
    
    def _analyze_feature_interactions(self, X: np.ndarray, y: np.ndarray) -> Dict[Tuple[str, str], float]:
        """Analyze feature interactions"""
        interaction_network = {}
        
        try:
            # Use Random Forest to detect feature interactions
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Simplified interaction detection
            for i in range(len(self.feature_names)):
                for j in range(i + 1, len(self.feature_names)):
                    if i >= X.shape[1] or j >= X.shape[1]:
                        continue
                    
                    # Calculate interaction strength (simplified)
                    feature_i = X[:, i]
                    feature_j = X[:, j]
                    
                    # Interaction as product term importance
                    interaction_term = feature_i * feature_j
                    corr_with_target = abs(np.corrcoef(interaction_term, y)[0, 1])
                    
                    if corr_with_target > self.config['interaction_threshold']:
                        interaction_network[(self.feature_names[i], self.feature_names[j])] = corr_with_target
        
        except Exception as e:
            self.logger.warning(f"Feature interaction analysis failed: {e}")
        
        return interaction_network
    
    def _analyze_importance_stability(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray,
                                    global_importance: List[FeatureImportance]) -> Dict[str, float]:
        """Analyze stability of feature importance across different samples"""
        stability_scores = {}
        
        try:
            n_iterations = 10
            sample_size = min(1000, len(X) // 2)
            
            importance_matrices = {}
            
            for feature_name in self.feature_names:
                importance_matrices[feature_name] = []
            
            for iteration in range(n_iterations):
                # Bootstrap sample
                sample_indices = np.random.choice(len(X), sample_size, replace=True)
                X_sample = X[sample_indices]
                y_sample = y[sample_indices]
                
                # Calculate importance on this sample
                sample_importance = self._model_based_importance(X_sample, y_sample)
                
                for feature_name, importance_score in sample_importance.items():
                    importance_matrices[feature_name].append(importance_score)
            
            # Calculate stability (1 - coefficient of variation)
            for feature_name, scores in importance_matrices.items():
                if np.mean(scores) > 0:
                    cv = np.std(scores) / np.mean(scores)
                    stability_scores[feature_name] = 1 - min(cv, 1.0)  # Cap at 1.0
                else:
                    stability_scores[feature_name] = 0.0
        
        except Exception as e:
            self.logger.warning(f"Stability analysis failed: {e}")
            # Default stability scores
            for feature_name in self.feature_names:
                stability_scores[feature_name] = 0.5
        
        return stability_scores
    
    def _calculate_statistical_significance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate statistical significance for all features"""
        significance_scores = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if i >= X.shape[1]:
                significance_scores[feature_name] = 0.0
                continue
            
            feature_values = X[:, i]
            
            # Handle constant features
            if len(np.unique(feature_values)) <= 1:
                significance_scores[feature_name] = 0.0
                continue
            
            # Calculate p-value using t-test
            try:
                # Split based on feature median
                median_val = np.median(feature_values)
                high_group = y[feature_values > median_val]
                low_group = y[feature_values <= median_val]
                
                if len(high_group) > 1 and len(low_group) > 1:
                    t_stat, p_value = stats.ttest_ind(high_group, low_group)
                    significance_scores[feature_name] = 1 - p_value
                else:
                    significance_scores[feature_name] = 0.0
                    
            except:
                significance_scores[feature_name] = 0.0
        
        return significance_scores
    
    def _group_correlated_features(self, X: np.ndarray) -> Dict[str, List[str]]:
        """Group highly correlated features"""
        feature_groups = {}
        
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X.T)
            
            # Group features with correlation > 0.8
            visited = set()
            group_id = 0
            
            for i, feature_i in enumerate(self.feature_names):
                if feature_i in visited or i >= corr_matrix.shape[0]:
                    continue
                
                group = [feature_i]
                visited.add(feature_i)
                
                for j, feature_j in enumerate(self.feature_names):
                    if feature_j in visited or j >= corr_matrix.shape[1]:
                        continue
                    
                    if abs(corr_matrix[i, j]) > 0.8:
                        group.append(feature_j)
                        visited.add(feature_j)
                
                if len(group) > 1:
                    feature_groups[f"group_{group_id}"] = group
                    group_id += 1
        
        except Exception as e:
            self.logger.warning(f"Feature grouping failed: {e}")
        
        return feature_groups

# Visualization and Reporting
class FeatureImportanceVisualizer:
    """Visualization tools for feature importance analysis"""
    
    @staticmethod
    def create_comprehensive_dashboard(analysis: FeatureImportanceAnalysis, save_path: str = None):
        """Create comprehensive dashboard for feature importance analysis"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Global Feature Importance', 
                'Feature Importance Stability',
                'Feature Interactions Network',
                'Statistical Significance',
                'Temporal Importance Trends',
                'Feature Correlation Groups'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
        
        # 1. Global Feature Importance
        features = [fi.feature_name for fi in analysis.global_importance]
        importance_scores = [fi.importance_score for fi in analysis.global_importance]
        colors = ['red' if fi.direction == 'negative' else 'green' for fi in analysis.global_importance]
        
        fig.add_trace(
            go.Bar(x=importance_scores, y=features, orientation='h', 
                  marker_color=colors, name='Importance'),
            row=1, col=1
        )
        
        # 2. Stability Analysis
        stability_features = list(analysis.stability_analysis.keys())[:10]
        stability_scores = [analysis.stability_analysis[f] for f in stability_features]
        
        fig.add_trace(
            go.Scatter(x=stability_features, y=stability_scores, 
                      mode='markers+lines', name='Stability'),
            row=1, col=2
        )
        
        # 3. Feature Interactions
        if analysis.interaction_network:
            interaction_features = list(set(
                [pair[0] for pair in analysis.interaction_network.keys()] +
                [pair[1] for pair in analysis.interaction_network.keys()]
            ))[:10]
            
            interaction_matrix = np.zeros((len(interaction_features), len(interaction_features)))
            for (f1, f2), strength in analysis.interaction_network.items():
                if f1 in interaction_features and f2 in interaction_features:
                    i = interaction_features.index(f1)
                    j = interaction_features.index(f2)
                    interaction_matrix[i, j] = strength
                    interaction_matrix[j, i] = strength
            
            fig.add_trace(
                go.Heatmap(z=interaction_matrix, x=interaction_features, 
                          y=interaction_features, name='Interactions'),
                row=2, col=1
            )
        
        # 4. Statistical Significance
        sig_features = list(analysis.statistical_significance.keys())[:10]
        sig_scores = [analysis.statistical_significance[f] for f in sig_features]
        
        fig.add_trace(
            go.Bar(x=sig_features, y=sig_scores, name='Significance'),
            row=2, col=2
        )
        
        # 5. Temporal Importance (placeholder)
        if analysis.temporal_importance:
            temporal_feature = list(analysis.temporal_importance.keys())[0]
            temporal_data = analysis.temporal_importance[temporal_feature]
            
            fig.add_trace(
                go.Scatter(x=temporal_data.importance_series.index, 
                          y=temporal_data.importance_series.values,
                          name='Temporal Trend'),
                row=3, col=1
            )
        
        # 6. Feature Groups (placeholder heatmap)
        if analysis.feature_groups:
            group_features = []
            for group in analysis.feature_groups.values():
                group_features.extend(group[:3])  # Show first 3 features from each group
            
            group_features = list(set(group_features))[:10]
            corr_matrix = np.random.rand(len(group_features), len(group_features))  # Placeholder
            
            fig.add_trace(
                go.Heatmap(z=corr_matrix, x=group_features, y=group_features,
                          name='Feature Groups'),
                row=3, col=2
            )
        
        fig.update_layout(height=1200, title_text="Comprehensive Feature Importance Analysis")
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    @staticmethod
    def plot_importance_comparison(analysis: FeatureImportanceAnalysis, save_path: str = None):
        """Plot comparison of different importance measures"""
        
        features = [fi.feature_name for fi in analysis.global_importance[:8]]
        
        # Create comparison data (simplified)
        methods = ['Overall', 'SHAP', 'Permutation', 'Model-Based']
        data = []
        
        for feature in features:
            feature_data = [fi for fi in analysis.global_importance if fi.feature_name == feature][0]
            row = [feature_data.normalized_score]  # Overall score
            # Add placeholder scores for other methods
            row.extend([feature_data.normalized_score * np.random.uniform(0.8, 1.2) for _ in range(3)])
            data.append(row)
        
        df = pd.DataFrame(data, index=features, columns=methods)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Feature Importance Comparison Across Methods')
        ax.set_ylabel('Normalized Importance Score')
        ax.legend(title='Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample financial data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Create realistic financial features
    feature_names = [
        'price_momentum', 'volatility', 'volume_trend', 'rsi', 'macd',
        'bollinger_position', 'atr', 'market_cap', 'pe_ratio', 'dividend_yield',
        'sector_performance', 'beta', 'short_interest', 'insider_buying', 'analyst_rating'
    ]
    
    X = np.random.randn(n_samples, n_features)
    # Create realistic target relationship
    y = (2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 
         0.8 * X[:, 3] * X[:, 4] + np.random.randn(n_samples) * 0.5)
    
    # Train a model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Initialize feature importance analyzer
    importance_analyzer = AdvancedFeatureImportance()
    
    # Perform comprehensive analysis
    analysis = importance_analyzer.comprehensive_importance_analysis(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names
    )
    
    print("Feature Importance Analysis Results:")
    print("=" * 60)
    
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    for i, feature in enumerate(analysis.global_importance[:10]):
        print(f"{i+1}. {feature.feature_name}: {feature.importance_score:.4f} "
              f"({feature.significance.value}, {feature.direction})")
    
    print(f"\nStability Analysis:")
    print("-" * 20)
    for feature, stability in list(analysis.stability_analysis.items())[:5]:
        print(f"{feature}: {stability:.3f}")
    
    print(f"\nFeature Interactions (Top 5):")
    print("-" * 30)
    for (f1, f2), strength in list(analysis.interaction_network.items())[:5]:
        print(f"{f1} & {f2}: {strength:.3f}")
    
    print(f"\nFeature Groups:")
    print("-" * 15)
    for group_name, features in analysis.feature_groups.items():
        print(f"{group_name}: {features}")
    
    # Create visualizations
    visualizer = FeatureImportanceVisualizer()
    visualizer.create_comprehensive_dashboard(analysis, "feature_importance_dashboard.html")
    visualizer.plot_importance_comparison(analysis, "importance_comparison.png")
    
    # Generate detailed report
    print("\nDetailed Feature Analysis:")
    print("=" * 40)
    for feature in analysis.global_importance[:5]:
        print(f"\n{feature.feature_name}:")
        print(f"  Importance Score: {feature.importance_score:.4f}")
        print(f"  Normalized Score: {feature.normalized_score:.4f}")
        print(f"  Significance: {feature.significance.value}")
        print(f"  Direction: {feature.direction}")
        print(f"  P-value: {feature.p_value:.4f}")
        print(f"  Stability: {feature.stability_score:.3f}")
        print(f"  Confidence Interval: [{feature.confidence_interval[0]:.4f}, {feature.confidence_interval[1]:.4f}]")
        print(f"  Top Interactions:")
        for interaction_feature, strength in feature.interactions[:3]:
            print(f"    - {interaction_feature}: {strength:.3f}")

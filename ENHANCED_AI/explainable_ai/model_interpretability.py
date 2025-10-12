# model_interpretability.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import shap
import lime
import lime.lime_tabular
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class InterpretationMethod(Enum):
    SHAP = "shap"
    LIME = "lime"
    ATTENTION = "attention"
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    COUNTERFACTUAL = "counterfactual"

class ModelComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class FeatureImportance:
    feature_name: str
    importance_score: float
    direction: str  # positive, negative, neutral
    confidence: float
    statistical_significance: float

@dataclass
class ModelInterpretation:
    model_complexity: ModelComplexity
    global_importance: List[FeatureImportance]
    local_explanations: Dict[str, Any]
    decision_boundary: Dict[str, float]
    feature_interactions: List[Tuple[str, str, float]]
    model_confidence: float
    interpretation_confidence: float

@dataclass
class PredictionExplanation:
    prediction_value: float
    base_value: float
    feature_contributions: Dict[str, float]
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]
    confidence_interval: Tuple[float, float]
    counterfactual_suggestions: List[Dict[str, Any]]

class AdvancedModelInterpreter:
    """
    Advanced model interpretability system for financial AI models
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'shap_samples': 100,
            'lime_samples': 1000,
            'confidence_level': 0.95,
            'feature_interaction_threshold': 0.1,
            'max_display_features': 10,
            'statistical_test_alpha': 0.05
        }
        
        self.explainer = None
        self.feature_names = None
        self.model = None
        
    def interpret_model(self,
                       model: Any,
                       X_train: np.ndarray,
                       X_test: np.ndarray,
                       y_train: np.ndarray,
                       feature_names: List[str],
                       model_type: str = "transformer") -> ModelInterpretation:
        """
        Comprehensive model interpretation using multiple methods
        
        Args:
            model: Trained model to interpret
            X_train: Training features
            X_test: Test features for local explanations
            y_train: Training targets
            feature_names: List of feature names
            model_type: Type of model (transformer, tree, neural_network)
            
        Returns:
            ModelInterpretation object
        """
        
        self.model = model
        self.feature_names = feature_names
        
        try:
            # 1. Global feature importance
            global_importance = self._calculate_global_importance(X_train, y_train, model_type)
            
            # 2. Local explanations for sample predictions
            local_explanations = self._generate_local_explanations(X_test, model_type)
            
            # 3. Decision boundary analysis
            decision_boundary = self._analyze_decision_boundary(X_train, y_train)
            
            # 4. Feature interactions
            feature_interactions = self._analyze_feature_interactions(X_train, y_train)
            
            # 5. Model complexity assessment
            model_complexity = self._assess_model_complexity(model, X_train.shape[1])
            
            # 6. Confidence assessments
            model_confidence = self._calculate_model_confidence(X_test)
            interpretation_confidence = self._calculate_interpretation_confidence(
                global_importance, local_explanations
            )
            
            return ModelInterpretation(
                model_complexity=model_complexity,
                global_importance=global_importance,
                local_explanations=local_explanations,
                decision_boundary=decision_boundary,
                feature_interactions=feature_interactions,
                model_confidence=model_confidence,
                interpretation_confidence=interpretation_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error in model interpretation: {e}")
            raise
    
    def _calculate_global_importance(self, 
                                   X: np.ndarray, 
                                   y: np.ndarray, 
                                   model_type: str) -> List[FeatureImportance]:
        """Calculate global feature importance using multiple methods"""
        
        importance_scores = {}
        
        # Method 1: Permutation Importance
        if model_type in ["transformer", "neural_network"]:
            perm_importance = self._permutation_importance(X, y)
            importance_scores['permutation'] = perm_importance
        
        # Method 2: SHAP Global Importance
        if hasattr(self.model, 'predict'):
            shap_importance = self._shap_global_importance(X)
            importance_scores['shap'] = shap_importance
        
        # Method 3: Attention Weights (for transformers)
        if model_type == "transformer" and hasattr(self.model, 'attention_weights'):
            attention_importance = self._attention_based_importance(X)
            importance_scores['attention'] = attention_importance
        
        # Method 4: Statistical Correlation
        correlation_importance = self._statistical_correlation_importance(X, y)
        importance_scores['correlation'] = correlation_importance
        
        # Combine importance scores
        combined_importance = self._combine_importance_scores(importance_scores)
        
        # Create FeatureImportance objects
        feature_importances = []
        for feature_name, (score, direction, confidence) in combined_importance.items():
            # Statistical significance test
            stat_significance = self._calculate_statistical_significance(
                X, y, feature_name, self.feature_names.index(feature_name)
            )
            
            feature_importance = FeatureImportance(
                feature_name=feature_name,
                importance_score=score,
                direction=direction,
                confidence=confidence,
                statistical_significance=stat_significance
            )
            feature_importances.append(feature_importance)
        
        # Sort by importance score
        feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
        
        return feature_importances[:self.config['max_display_features']]
    
    def _permutation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate permutation importance"""
        try:
            # Use scikit-learn's permutation importance
            if hasattr(self.model, 'predict'):
                perm_importance = permutation_importance(
                    self.model, X, y, 
                    n_repeats=10,
                    random_state=42
                )
                
                importance_dict = {}
                for i, feature_name in enumerate(self.feature_names):
                    importance_dict[feature_name] = perm_importance.importances_mean[i]
                
                return importance_dict
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
        
        return {name: 0.0 for name in self.feature_names}
    
    def _shap_global_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate SHAP global importance"""
        try:
            # Sample data for SHAP computation
            if len(X) > self.config['shap_samples']:
                X_sample = X[np.random.choice(len(X), self.config['shap_samples'], replace=False)]
            else:
                X_sample = X
            
            # Initialize SHAP explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.KernelExplainer(self.model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            importance_dict = {}
            for i, feature_name in enumerate(self.feature_names):
                importance_dict[feature_name] = mean_abs_shap[i] if i < len(mean_abs_shap) else 0.0
            
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"SHAP importance failed: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def _attention_based_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate importance based on attention weights (for transformers)"""
        try:
            if not hasattr(self.model, 'attention_weights'):
                return {name: 0.0 for name in self.feature_names}
            
            # Get attention weights from model
            attention_weights = self.model.attention_weights
            
            if attention_weights is None:
                return {name: 0.0 for name in self.feature_names}
            
            # Average attention across heads and layers
            if len(attention_weights.shape) == 4:  # (batch, heads, seq, seq)
                avg_attention = np.mean(attention_weights, axis=(0, 1, 3))  # Average over batch, heads, and sequence
            else:
                avg_attention = np.mean(attention_weights, axis=0)
            
            # Map to features (simplified - assumes features correspond to sequence positions)
            importance_dict = {}
            n_features = min(len(self.feature_names), len(avg_attention))
            
            for i in range(n_features):
                importance_dict[self.feature_names[i]] = avg_attention[i]
            
            # Pad with zeros if needed
            for i in range(n_features, len(self.feature_names)):
                importance_dict[self.feature_names[i]] = 0.0
            
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"Attention-based importance failed: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def _statistical_correlation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate statistical correlation importance"""
        importance_dict = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if i >= X.shape[1]:
                importance_dict[feature_name] = 0.0
                continue
            
            feature_values = X[:, i]
            
            # Calculate correlation coefficient
            if len(np.unique(feature_values)) > 1:  # Avoid constant features
                correlation, p_value = stats.pearsonr(feature_values, y)
                importance_dict[feature_name] = abs(correlation) * (1 - p_value)
            else:
                importance_dict[feature_name] = 0.0
        
        return importance_dict
    
    def _combine_importance_scores(self, importance_scores: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, str, float]]:
        """Combine importance scores from multiple methods"""
        combined_scores = {}
        
        for feature_name in self.feature_names:
            scores = []
            directions = []
            confidences = []
            
            for method, scores_dict in importance_scores.items():
                if feature_name in scores_dict:
                    score = scores_dict[feature_name]
                    scores.append(score)
                    
                    # Determine direction (simplified)
                    direction = "positive" if score > 0 else "negative" if score < 0 else "neutral"
                    directions.append(direction)
                    
                    # Confidence based on method reliability
                    method_confidence = {
                        'shap': 0.9,
                        'permutation': 0.8,
                        'attention': 0.7,
                        'correlation': 0.6
                    }.get(method, 0.5)
                    confidences.append(method_confidence)
            
            if scores:
                # Weighted average based on confidence
                total_confidence = sum(confidences)
                if total_confidence > 0:
                    weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_confidence
                    avg_confidence = np.mean(confidences)
                    
                    # Most common direction
                    direction_counts = {d: directions.count(d) for d in set(directions)}
                    final_direction = max(direction_counts.items(), key=lambda x: x[1])[0]
                else:
                    weighted_score = np.mean(scores)
                    avg_confidence = 0.5
                    final_direction = "neutral"
            else:
                weighted_score = 0.0
                avg_confidence = 0.0
                final_direction = "neutral"
            
            combined_scores[feature_name] = (weighted_score, final_direction, avg_confidence)
        
        return combined_scores
    
    def _calculate_statistical_significance(self, 
                                          X: np.ndarray, 
                                          y: np.ndarray, 
                                          feature_name: str, 
                                          feature_idx: int) -> float:
        """Calculate statistical significance of feature"""
        try:
            if feature_idx >= X.shape[1]:
                return 1.0  # Not significant
            
            feature_values = X[:, feature_idx]
            
            # Simple t-test for feature significance
            if len(np.unique(feature_values)) > 1:
                # Split based on feature median
                median_val = np.median(feature_values)
                high_group = y[feature_values > median_val]
                low_group = y[feature_values <= median_val]
                
                if len(high_group) > 1 and len(low_group) > 1:
                    t_stat, p_value = stats.ttest_ind(high_group, low_group)
                    return 1 - p_value  # Convert to significance score
            
            return 0.5  # Moderate significance by default
            
        except Exception as e:
            self.logger.warning(f"Statistical significance calculation failed: {e}")
            return 0.5
    
    def _generate_local_explanations(self, X: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Generate local explanations for sample predictions"""
        local_explanations = {}
        
        # Sample a few instances for local explanation
        n_samples = min(5, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            instance = X[idx:idx+1]
            
            explanation = self._explain_single_prediction(instance, model_type)
            local_explanations[f"instance_{i}"] = {
                'prediction': explanation.prediction_value,
                'feature_contributions': explanation.feature_contributions,
                'top_positive': explanation.top_positive_features,
                'top_negative': explanation.top_negative_features,
                'confidence_interval': explanation.confidence_interval,
                'counterfactuals': explanation.counterfactual_suggestions
            }
        
        return local_explanations
    
    def _explain_single_prediction(self, 
                                 instance: np.ndarray, 
                                 model_type: str) -> PredictionExplanation:
        """Explain a single prediction"""
        
        # Get prediction
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(instance)[0]
        else:
            # For PyTorch models
            with torch.no_grad():
                instance_tensor = torch.FloatTensor(instance)
                prediction = self.model(instance_tensor).item()
        
        # Feature contributions using LIME
        feature_contributions = self._lime_explanation(instance)
        
        # Separate positive and negative contributions
        positive_contributions = [(feat, contrib) for feat, contrib in feature_contributions.items() if contrib > 0]
        negative_contributions = [(feat, contrib) for feat, contrib in feature_contributions.items() if contrib < 0]
        
        # Sort by absolute contribution
        positive_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        negative_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Calculate base value (mean prediction)
        base_value = prediction - sum(feature_contributions.values())
        
        # Confidence interval (simplified)
        confidence_interval = (
            prediction - 0.1 * abs(prediction),
            prediction + 0.1 * abs(prediction)
        )
        
        # Counterfactual suggestions
        counterfactuals = self._generate_counterfactuals(instance, prediction, feature_contributions)
        
        return PredictionExplanation(
            prediction_value=prediction,
            base_value=base_value,
            feature_contributions=feature_contributions,
            top_positive_features=positive_contributions[:3],
            top_negative_features=negative_contributions[:3],
            confidence_interval=confidence_interval,
            counterfactual_suggestions=counterfactuals
        )
    
    def _lime_explanation(self, instance: np.ndarray) -> Dict[str, float]:
        """Generate LIME explanation for a single instance"""
        try:
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.random.randn(100, instance.shape[1]),  # Dummy data
                feature_names=self.feature_names,
                mode='regression',
                random_state=42
            )
            
            # Explain instance
            exp = explainer.explain_instance(
                instance[0], 
                self.model.predict,
                num_features=len(self.feature_names)
            )
            
            feature_contributions = {}
            for feature, contribution in exp.as_list():
                feature_contributions[feature] = contribution
            
            return feature_contributions
            
        except Exception as e:
            self.logger.warning(f"LIME explanation failed: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def _generate_counterfactuals(self, 
                                instance: np.ndarray, 
                                prediction: float,
                                feature_contributions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        counterfactuals = []
        
        # Find the most influential negative feature
        negative_features = [(feat, contrib) for feat, contrib in feature_contributions.items() if contrib < 0]
        if negative_features:
            most_negative_feature, neg_contribution = max(negative_features, key=lambda x: abs(x[1]))
            
            counterfactual_1 = {
                'suggestion': f"Increase {most_negative_feature} to reduce negative impact",
                'expected_effect': f"Could improve prediction by {abs(neg_contribution):.4f}",
                'feature': most_negative_feature,
                'current_contribution': neg_contribution
            }
            counterfactuals.append(counterfactual_1)
        
        # Find the most influential positive feature
        positive_features = [(feat, contrib) for feat, contrib in feature_contributions.items() if contrib > 0]
        if positive_features:
            most_positive_feature, pos_contribution = max(positive_features, key=lambda x: abs(x[1]))
            
            counterfactual_2 = {
                'suggestion': f"Maintain or slightly increase {most_positive_feature}",
                'expected_effect': f"Preserves positive contribution of {pos_contribution:.4f}",
                'feature': most_positive_feature,
                'current_contribution': pos_contribution
            }
            counterfactuals.append(counterfactual_2)
        
        return counterfactuals
    
    def _analyze_decision_boundary(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Analyze model decision boundaries"""
        try:
            # Simple decision boundary analysis using feature ranges
            boundary_analysis = {}
            
            for i, feature_name in enumerate(self.feature_names):
                if i >= X.shape[1]:
                    continue
                
                feature_values = X[:, i]
                boundary_analysis[feature_name] = {
                    'range': (np.min(feature_values), np.max(feature_values)),
                    'std': np.std(feature_values),
                    'decision_threshold': np.median(feature_values)  # Simplified
                }
            
            return boundary_analysis
            
        except Exception as e:
            self.logger.warning(f"Decision boundary analysis failed: {e}")
            return {}
    
    def _analyze_feature_interactions(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[str, str, float]]:
        """Analyze interactions between features"""
        interactions = []
        
        try:
            # Use Random Forest to detect feature interactions
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Simplified interaction detection
            # In practice, use more sophisticated methods like H-statistic
            for i in range(len(self.feature_names)):
                for j in range(i + 1, len(self.feature_names)):
                    if i >= X.shape[1] or j >= X.shape[1]:
                        continue
                    
                    # Simple correlation between features as interaction proxy
                    corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    interaction_strength = abs(corr)
                    
                    if interaction_strength > self.config['feature_interaction_threshold']:
                        interactions.append((
                            self.feature_names[i],
                            self.feature_names[j],
                            interaction_strength
                        ))
            
            # Sort by interaction strength
            interactions.sort(key=lambda x: x[2], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Feature interaction analysis failed: {e}")
        
        return interactions[:10]  # Return top 10 interactions
    
    def _assess_model_complexity(self, model: Any, n_features: int) -> ModelComplexity:
        """Assess model complexity"""
        try:
            # Count parameters for neural networks
            if hasattr(model, 'parameters'):
                n_params = sum(p.numel() for p in model.parameters())
            else:
                n_params = 0
            
            # Simple complexity assessment
            complexity_score = n_params / (n_features + 1)  # Normalize by features
            
            if complexity_score > 1000:
                return ModelComplexity.HIGH
            elif complexity_score > 100:
                return ModelComplexity.MEDIUM
            else:
                return ModelComplexity.LOW
                
        except Exception as e:
            self.logger.warning(f"Model complexity assessment failed: {e}")
            return ModelComplexity.MEDIUM
    
    def _calculate_model_confidence(self, X: np.ndarray) -> float:
        """Calculate overall model confidence"""
        try:
            if hasattr(self.model, 'predict_proba'):
                # For classifiers, use prediction probabilities
                predictions = self.model.predict_proba(X)
                confidence = np.mean(np.max(predictions, axis=1))
            else:
                # For regressors, use prediction stability
                # Multiple predictions with slight perturbations
                n_samples = min(10, len(X))
                sample_indices = np.random.choice(len(X), n_samples, replace=False)
                
                stability_scores = []
                for idx in sample_indices:
                    instance = X[idx:idx+1]
                    predictions = []
                    
                    # Multiple slight perturbations
                    for _ in range(5):
                        perturbed = instance + np.random.normal(0, 0.01, instance.shape)
                        if hasattr(self.model, 'predict'):
                            pred = self.model.predict(perturbed)[0]
                        else:
                            with torch.no_grad():
                                pred = self.model(torch.FloatTensor(perturbed)).item()
                        predictions.append(pred)
                    
                    # Stability = 1 - coefficient of variation
                    if np.std(predictions) > 0:
                        stability = 1 - (np.std(predictions) / np.mean(predictions))
                        stability_scores.append(max(0, stability))
                    else:
                        stability_scores.append(1.0)
                
                confidence = np.mean(stability_scores)
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"Model confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_interpretation_confidence(self,
                                           global_importance: List[FeatureImportance],
                                           local_explanations: Dict[str, Any]) -> float:
        """Calculate confidence in interpretation results"""
        try:
            # Based on consistency between global and local explanations
            consistency_scores = []
            
            for instance_key, local_exp in local_explanations.items():
                local_top_features = set([feat for feat, _ in local_exp['top_positive'][:3] + local_exp['top_negative'][:3]])
                global_top_features = set([feat.feature_name for feat in global_importance[:5]])
                
                overlap = len(local_top_features.intersection(global_top_features))
                consistency = overlap / len(local_top_features.union(global_top_features)) if local_top_features else 0.5
                consistency_scores.append(consistency)
            
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.5
            
            # Based on feature importance confidence
            avg_feature_confidence = np.mean([feat.confidence for feat in global_importance]) if global_importance else 0.5
            
            # Combined confidence
            interpretation_confidence = (avg_consistency + avg_feature_confidence) / 2
            
            return interpretation_confidence
            
        except Exception as e:
            self.logger.warning(f"Interpretation confidence calculation failed: {e}")
            return 0.5
    
    def generate_interpretation_report(self, interpretation: ModelInterpretation) -> Dict[str, Any]:
        """Generate comprehensive interpretation report"""
        
        report = {
            'summary': {
                'model_complexity': interpretation.model_complexity.value,
                'overall_confidence': interpretation.model_confidence,
                'interpretation_reliability': interpretation.interpretation_confidence,
                'key_insights': self._extract_key_insights(interpretation)
            },
            'feature_analysis': {
                'top_features': [
                    {
                        'feature': feat.feature_name,
                        'importance': feat.importance_score,
                        'direction': feat.direction,
                        'confidence': feat.confidence,
                        'significance': feat.statistical_significance
                    }
                    for feat in interpretation.global_importance
                ],
                'feature_interactions': [
                    {
                        'feature_1': interaction[0],
                        'feature_2': interaction[1],
                        'strength': interaction[2]
                    }
                    for interaction in interpretation.feature_interactions
                ]
            },
            'local_explanations': interpretation.local_explanations,
            'decision_boundaries': interpretation.decision_boundary,
            'recommendations': self._generate_recommendations(interpretation)
        }
        
        return report
    
    def _extract_key_insights(self, interpretation: ModelInterpretation) -> List[str]:
        """Extract key insights from interpretation results"""
        insights = []
        
        # Top feature insights
        if interpretation.global_importance:
            top_feature = interpretation.global_importance[0]
            insights.append(
                f"Most important feature: {top_feature.feature_name} "
                f"({top_feature.direction} impact, confidence: {top_feature.confidence:.2f})"
            )
        
        # Model complexity insight
        insights.append(f"Model complexity: {interpretation.model_complexity.value}")
        
        # Interaction insights
        if interpretation.feature_interactions:
            strongest_interaction = interpretation.feature_interactions[0]
            insights.append(
                f"Strongest feature interaction: {strongest_interaction[0]} & "
                f"{strongest_interaction[1]} (strength: {strongest_interaction[2]:.2f})"
            )
        
        # Confidence insight
        insights.append(f"Model confidence: {interpretation.model_confidence:.2f}")
        insights.append(f"Interpretation reliability: {interpretation.interpretation_confidence:.2f}")
        
        return insights
    
    def _generate_recommendations(self, interpretation: ModelInterpretation) -> List[str]:
        """Generate actionable recommendations based on interpretation"""
        recommendations = []
        
        # Feature-related recommendations
        if interpretation.global_importance:
            top_features = [feat.feature_name for feat in interpretation.global_importance[:3]]
            recommendations.append(
                f"Focus on monitoring and engineering these key features: {', '.join(top_features)}"
            )
        
        # Model complexity recommendations
        if interpretation.model_complexity == ModelComplexity.HIGH:
            recommendations.append(
                "Consider model simplification or regularization to improve interpretability"
            )
        elif interpretation.model_complexity == ModelComplexity.LOW:
            recommendations.append(
                "Model may be underfitting - consider increasing complexity for better performance"
            )
        
        # Confidence-based recommendations
        if interpretation.model_confidence < 0.7:
            recommendations.append(
                "Model predictions have moderate uncertainty - use with caution in high-stakes decisions"
            )
        
        if interpretation.interpretation_confidence < 0.7:
            recommendations.append(
                "Interpretation reliability is moderate - consider additional validation of feature importance"
            )
        
        # Feature interaction recommendations
        if interpretation.feature_interactions:
            recommendations.append(
                "Account for feature interactions in decision-making process"
            )
        
        return recommendations

# Example usage with visualization
class InterpretationVisualizer:
    """Visualization tools for model interpretation"""
    
    @staticmethod
    def plot_feature_importance(interpretation: ModelInterpretation, save_path: str = None):
        """Plot feature importance with confidence intervals"""
        features = [feat.feature_name for feat in interpretation.global_importance]
        importance_scores = [feat.importance_score for feat in interpretation.global_importance]
        confidences = [feat.confidence for feat in interpretation.global_importance]
        colors = ['green' if feat.direction == 'positive' else 'red' if feat.direction == 'negative' else 'blue' 
                 for feat in interpretation.global_importance]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, importance_scores, color=colors, alpha=0.7)
        
        # Add confidence indicators
        for i, (score, conf) in enumerate(zip(importance_scores, confidences)):
            ax.text(score + 0.01, i, f'{conf:.2f}', va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance Score')
        ax.set_title('Feature Importance with Confidence Scores')
        ax.legend(['Positive', 'Negative', 'Neutral'])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_interactive_dashboard(interpretation: ModelInterpretation, report: Dict[str, Any]):
        """Create interactive dashboard for model interpretation"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance', 'Feature Interactions', 
                          'Model Confidence', 'Local Explanations'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "indicator"}, {"type": "table"}]]
        )
        
        # Feature Importance
        features = [feat.feature_name for feat in interpretation.global_importance]
        importance_scores = [feat.importance_score for feat in interpretation.global_importance]
        
        fig.add_trace(
            go.Bar(x=importance_scores, y=features, orientation='h', name='Importance'),
            row=1, col=1
        )
        
        # Feature Interactions (simplified heatmap)
        interaction_matrix = np.zeros((len(features), len(features)))
        for interaction in interpretation.feature_interactions:
            try:
                i = features.index(interaction[0])
                j = features.index(interaction[1])
                interaction_matrix[i, j] = interaction[2]
                interaction_matrix[j, i] = interaction[2]
            except ValueError:
                continue
        
        fig.add_trace(
            go.Heatmap(z=interaction_matrix, x=features, y=features, name='Interactions'),
            row=1, col=2
        )
        
        # Model Confidence
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=interpretation.model_confidence,
                title={'text': "Model Confidence"},
                domain={'row': 1, 'col': 1},
                gauge={'axis': {'range': [0, 1]}}
            ),
            row=2, col=1
        )
        
        # Local Explanations Table
        local_data = []
        for instance_key, local_exp in interpretation.local_explanations.items():
            top_features = [feat for feat, _ in local_exp['top_positive'][:2] + local_exp['top_negative'][:2]]
            contributions = [contrib for _, contrib in local_exp['top_positive'][:2] + local_exp['top_negative'][:2]]
            local_data.append([instance_key] + top_features + [f"{c:.4f}" for c in contributions])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Instance', 'Feature 1', 'Feature 2', 'Contrib 1', 'Contrib 2']),
                cells=dict(values=[[row[i] for row in local_data] for i in range(5)])
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Model Interpretation Dashboard")
        fig.show()

# Example usage
if __name__ == "__main__":
    # Sample usage with dummy data
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Train a simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Initialize interpreter
    interpreter = AdvancedModelInterpreter()
    
    # Interpret model
    interpretation = interpreter.interpret_model(
        model=model,
        X_train=X,
        X_test=X[:10],
        y_train=y,
        feature_names=feature_names,
        model_type="tree"
    )
    
    # Generate report
    report = interpreter.generate_interpretation_report(interpretation)
    
    print("Model Interpretation Report:")
    print("=" * 50)
    print(f"Model Complexity: {interpretation.model_complexity.value}")
    print(f"Model Confidence: {interpretation.model_confidence:.2f}")
    print(f"Interpretation Reliability: {interpretation.interpretation_confidence:.2f}")
    
    print("\nTop Features:")
    for i, feat in enumerate(interpretation.global_importance[:5]):
        print(f"{i+1}. {feat.feature_name}: {feat.importance_score:.4f} "
              f"({feat.direction}, conf: {feat.confidence:.2f})")
    
    print("\nKey Insights:")
    for insight in report['summary']['key_insights']:
        print(f"- {insight}")
    
    print("\nRecommendations:")
    for recommendation in report['recommendations']:
        print(f"- {recommendation}")
    
    # Create visualizations
    visualizer = InterpretationVisualizer()
    visualizer.plot_feature_importance(interpretation)
    visualizer.create_interactive_dashboard(interpretation, report)

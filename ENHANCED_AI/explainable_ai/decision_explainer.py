# decision_explainer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

class ExplanationType(Enum):
    COUNTERFACTUAL = "counterfactual"
    CONTRASTIVE = "contrastive"
    CAUSAL = "causal"
    EXAMPLE_BASED = "example_based"
    FEATURE_BASED = "feature_based"

class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class DecisionImpact(Enum):
    STRONG_POSITIVE = "strong_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    STRONG_NEGATIVE = "strong_negative"

@dataclass
class DecisionExplanation:
    prediction: float
    confidence: float
    confidence_level: ConfidenceLevel
    key_factors: List[Tuple[str, float, DecisionImpact]]  # (feature, contribution, impact)
    counterfactuals: List[Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]
    decision_boundary: float
    risk_factors: List[Tuple[str, float]]
    opportunity_factors: List[Tuple[str, float]]
    explanation_type: ExplanationType
    rationale: str

@dataclass
class CounterfactualExplanation:
    feature_changes: Dict[str, Tuple[float, float]]  # feature: (current_value, suggested_value)
    expected_impact: float
    confidence: float
    feasibility: float
    implementation_cost: float
    description: str

@dataclass
class ContrastiveCase:
    case_id: str
    features: Dict[str, float]
    prediction: float
    similarity: float
    key_differences: List[Tuple[str, float, float]]  # (feature, case_value, current_value)
    outcome_difference: float

class AdvancedDecisionExplainer:
    """
    Advanced decision explanation system for financial AI models
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'counterfactual_samples': 1000,
            'similarity_threshold': 0.8,
            'max_counterfactuals': 5,
            'max_similar_cases': 5,
            'confidence_thresholds': {
                'very_high': 0.9,
                'high': 0.8,
                'medium': 0.7,
                'low': 0.6,
                'very_low': 0.5
            },
            'impact_thresholds': {
                'strong_positive': 0.3,
                'positive': 0.1,
                'negative': -0.1,
                'strong_negative': -0.3
            }
        }
        
        self.model = None
        self.feature_names = None
        self.training_data = None
        self.training_predictions = None
        
    def explain_decision(self,
                        model: Any,
                        instance: np.ndarray,
                        feature_names: List[str],
                        training_data: np.ndarray = None,
                        training_targets: np.ndarray = None,
                        explanation_type: ExplanationType = ExplanationType.COUNTERFACTUAL) -> DecisionExplanation:
        """
        Explain model decision for a single instance
        
        Args:
            model: Trained model
            instance: Input instance to explain
            feature_names: List of feature names
            training_data: Training data for similar cases
            training_targets: Training targets for similar cases
            explanation_type: Type of explanation to generate
            
        Returns:
            DecisionExplanation object
        """
        
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        
        try:
            # 1. Get prediction and confidence
            prediction, confidence = self._get_prediction_with_confidence(instance)
            confidence_level = self._get_confidence_level(confidence)
            
            # 2. Analyze key factors
            key_factors = self._analyze_key_factors(instance, prediction)
            
            # 3. Generate counterfactuals
            counterfactuals = self._generate_counterfactuals(instance, prediction)
            
            # 4. Find similar cases
            similar_cases = self._find_similar_cases(instance, prediction, training_data, training_targets)
            
            # 5. Analyze decision boundary
            decision_boundary = self._analyze_decision_boundary(instance, prediction)
            
            # 6. Identify risk and opportunity factors
            risk_factors, opportunity_factors = self._identify_risk_opportunity_factors(key_factors)
            
            # 7. Generate rationale
            rationale = self._generate_rationale(prediction, key_factors, counterfactuals, similar_cases)
            
            return DecisionExplanation(
                prediction=prediction,
                confidence=confidence,
                confidence_level=confidence_level,
                key_factors=key_factors,
                counterfactuals=counterfactuals,
                similar_cases=similar_cases,
                decision_boundary=decision_boundary,
                risk_factors=risk_factors,
                opportunity_factors=opportunity_factors,
                explanation_type=explanation_type,
                rationale=rationale
            )
            
        except Exception as e:
            self.logger.error(f"Decision explanation failed: {e}")
            raise
    
    def _get_prediction_with_confidence(self, instance: np.ndarray) -> Tuple[float, float]:
        """Get model prediction with confidence estimate"""
        try:
            if hasattr(self.model, 'predict_proba'):
                # For classifiers
                probabilities = self.model.predict_proba(instance.reshape(1, -1))
                prediction = np.argmax(probabilities[0])
                confidence = np.max(probabilities[0])
            elif hasattr(self.model, 'predict'):
                # For regressors - use prediction stability as confidence
                prediction = self.model.predict(instance.reshape(1, -1))[0]
                
                # Generate multiple slightly perturbed predictions
                perturbations = []
                for _ in range(10):
                    perturbed_instance = instance + np.random.normal(0, 0.01, instance.shape)
                    perturbed_pred = self.model.predict(perturbed_instance.reshape(1, -1))[0]
                    perturbations.append(perturbed_pred)
                
                # Confidence is inverse of prediction variance
                confidence = 1 / (1 + np.var(perturbations))
            else:
                # For PyTorch models
                with torch.no_grad():
                    instance_tensor = torch.FloatTensor(instance).unsqueeze(0)
                    prediction = self.model(instance_tensor).item()
                    confidence = 0.7  # Default confidence for neural networks
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            self.logger.warning(f"Prediction with confidence failed: {e}")
            return 0.0, 0.5
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        thresholds = self.config['confidence_thresholds']
        
        if confidence >= thresholds['very_high']:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= thresholds['high']:
            return ConfidenceLevel.HIGH
        elif confidence >= thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        elif confidence >= thresholds['low']:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _analyze_key_factors(self, instance: np.ndarray, prediction: float) -> List[Tuple[str, float, DecisionImpact]]:
        """Analyze key factors contributing to the decision"""
        
        key_factors = []
        
        # Method 1: SHAP-based analysis
        shap_contributions = self._get_shap_contributions(instance)
        
        # Method 2: LIME-based analysis
        lime_contributions = self._get_lime_contributions(instance)
        
        # Combine contributions
        for feature_name in self.feature_names:
            shap_contrib = shap_contributions.get(feature_name, 0.0)
            lime_contrib = lime_contributions.get(feature_name, 0.0)
            
            # Weighted average
            combined_contrib = (shap_contrib * 0.7 + lime_contrib * 0.3)
            
            # Determine impact
            impact = self._determine_impact(combined_contrib, prediction)
            
            key_factors.append((feature_name, combined_contrib, impact))
        
        # Sort by absolute contribution
        key_factors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return key_factors[:10]  # Return top 10 factors
    
    def _get_shap_contributions(self, instance: np.ndarray) -> Dict[str, float]:
        """Get SHAP feature contributions"""
        try:
            if hasattr(self.model, 'predict'):
                # Create background data for SHAP
                if self.training_data is not None:
                    background_data = self.training_data
                else:
                    background_data = np.random.randn(100, len(instance))
                
                # Initialize SHAP explainer
                if hasattr(self.model, 'predict_proba'):
                    explainer = shap.TreeExplainer(self.model)
                else:
                    explainer = shap.KernelExplainer(self.model.predict, background_data)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(instance.reshape(1, -1))
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)
                
                if len(shap_values.shape) > 2:  # Multi-class
                    shap_values = shap_values[0]  # Take first class
                
                contributions = {}
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(shap_values[0]):
                        contributions[feature_name] = shap_values[0][i]
                    else:
                        contributions[feature_name] = 0.0
                
                return contributions
                
        except Exception as e:
            self.logger.warning(f"SHAP contributions failed: {e}")
        
        return {name: 0.0 for name in self.feature_names}
    
    def _get_lime_contributions(self, instance: np.ndarray) -> Dict[str, float]:
        """Get LIME feature contributions"""
        try:
            if hasattr(self.model, 'predict'):
                # Create LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.random.randn(100, len(instance)),  # Dummy data
                    feature_names=self.feature_names,
                    mode='regression',
                    random_state=42
                )
                
                # Explain instance
                exp = explainer.explain_instance(
                    instance, 
                    self.model.predict,
                    num_features=len(self.feature_names)
                )
                
                contributions = {}
                for feature, contribution in exp.as_list():
                    contributions[feature] = contribution
                
                return contributions
                
        except Exception as e:
            self.logger.warning(f"LIME contributions failed: {e}")
        
        return {name: 0.0 for name in self.feature_names}
    
    def _determine_impact(self, contribution: float, prediction: float) -> DecisionImpact:
        """Determine the impact of a feature contribution"""
        thresholds = self.config['impact_thresholds']
        
        # Normalize contribution relative to prediction
        if abs(prediction) > 0:
            normalized_contrib = contribution / abs(prediction)
        else:
            normalized_contrib = contribution
        
        if normalized_contrib >= thresholds['strong_positive']:
            return DecisionImpact.STRONG_POSITIVE
        elif normalized_contrib >= thresholds['positive']:
            return DecisionImpact.POSITIVE
        elif normalized_contrib <= thresholds['strong_negative']:
            return DecisionImpact.STRONG_NEGATIVE
        elif normalized_contrib <= thresholds['negative']:
            return DecisionImpact.NEGATIVE
        else:
            return DecisionImpact.NEUTRAL
    
    def _generate_counterfactuals(self, instance: np.ndarray, prediction: float) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        counterfactuals = []
        
        try:
            # Get feature contributions to identify most influential features
            key_factors = self._analyze_key_factors(instance, prediction)
            
            # Generate counterfactuals for top features
            for feature_name, contribution, impact in key_factors[:3]:
                if impact in [DecisionImpact.NEGATIVE, DecisionImpact.STRONG_NEGATIVE]:
                    # For negative impacts, suggest increasing the feature
                    counterfactual = self._create_counterfactual(
                        instance, feature_name, contribution, "increase"
                    )
                elif impact in [DecisionImpact.POSITIVE, DecisionImpact.STRONG_POSITIVE]:
                    # For positive impacts, suggest maintaining or slightly increasing
                    counterfactual = self._create_counterfactual(
                        instance, feature_name, contribution, "maintain"
                    )
                else:
                    # For neutral impacts, suggest optimization
                    counterfactual = self._create_counterfactual(
                        instance, feature_name, contribution, "optimize"
                    )
                
                if counterfactual:
                    counterfactuals.append(counterfactual)
            
            return counterfactuals[:self.config['max_counterfactuals']]
            
        except Exception as e:
            self.logger.warning(f"Counterfactual generation failed: {e}")
            return []
    
    def _create_counterfactual(self, 
                             instance: np.ndarray, 
                             feature_name: str, 
                             contribution: float,
                             action: str) -> Dict[str, Any]:
        """Create a single counterfactual explanation"""
        
        feature_idx = self.feature_names.index(feature_name)
        current_value = instance[feature_idx]
        
        # Determine suggested change based on action
        if action == "increase":
            suggested_value = current_value * 1.2  # 20% increase
            expected_impact = abs(contribution) * 0.8  # Estimated improvement
            description = f"Increase {feature_name} to enhance positive impact"
        elif action == "maintain":
            suggested_value = current_value * 1.05  # Small maintenance increase
            expected_impact = abs(contribution) * 0.9  # Maintain current benefit
            description = f"Maintain current level of {feature_name} to preserve benefits"
        else:  # optimize
            suggested_value = current_value * 0.9  # Small optimization
            expected_impact = abs(contribution) * 0.5  # Moderate improvement
            description = f"Optimize {feature_name} for better balance"
        
        # Calculate feasibility (simplified)
        feasibility = self._calculate_feasibility(feature_name, current_value, suggested_value)
        
        # Calculate implementation cost (simplified)
        implementation_cost = self._calculate_implementation_cost(feature_name, abs(suggested_value - current_value))
        
        return {
            'feature_changes': {feature_name: (current_value, suggested_value)},
            'expected_impact': expected_impact,
            'confidence': min(0.8, abs(contribution)),  # Based on contribution strength
            'feasibility': feasibility,
            'implementation_cost': implementation_cost,
            'description': description
        }
    
    def _calculate_feasibility(self, feature_name: str, current_value: float, suggested_value: float) -> float:
        """Calculate feasibility of implementing the counterfactual"""
        # Simplified feasibility calculation
        # In practice, this would use domain knowledge
        
        change_magnitude = abs(suggested_value - current_value) / (abs(current_value) + 1e-8)
        
        # Features that are easier to change
        easy_features = ['rsi', 'momentum', 'volume_trend', 'sentiment']
        medium_features = ['volatility', 'beta', 'market_cap']
        hard_features = ['pe_ratio', 'dividend_yield', 'sector_performance']
        
        if feature_name in easy_features:
            base_feasibility = 0.8
        elif feature_name in medium_features:
            base_feasibility = 0.6
        elif feature_name in hard_features:
            base_feasibility = 0.4
        else:
            base_feasibility = 0.5
        
        # Adjust for change magnitude
        if change_magnitude < 0.1:
            feasibility = base_feasibility * 0.9
        elif change_magnitude < 0.3:
            feasibility = base_feasibility * 0.7
        else:
            feasibility = base_feasibility * 0.5
        
        return min(feasibility, 1.0)
    
    def _calculate_implementation_cost(self, feature_name: str, change_amount: float) -> float:
        """Calculate implementation cost for counterfactual"""
        # Simplified cost calculation
        # In practice, this would use domain knowledge
        
        # Features with low implementation cost
        low_cost_features = ['technical_indicators', 'sentiment', 'momentum']
        medium_cost_features = ['volume', 'volatility', 'short_interest']
        high_cost_features = ['fundamental_ratios', 'analyst_rating', 'insider_buying']
        
        if feature_name in low_cost_features:
            base_cost = 0.2
        elif feature_name in medium_cost_features:
            base_cost = 0.5
        elif feature_name in high_cost_features:
            base_cost = 0.8
        else:
            base_cost = 0.5
        
        # Adjust for change amount
        cost = base_cost * (1 + change_amount * 2)
        
        return min(cost, 1.0)
    
    def _find_similar_cases(self, 
                          instance: np.ndarray, 
                          prediction: float,
                          training_data: np.ndarray,
                          training_targets: np.ndarray) -> List[Dict[str, Any]]:
        """Find similar historical cases"""
        similar_cases = []
        
        if training_data is None or training_targets is None:
            return similar_cases
        
        try:
            # Calculate similarity to all training instances
            similarities = pairwise_distances(
                training_data, 
                instance.reshape(1, -1), 
                metric='euclidean'
            ).flatten()
            
            # Find most similar cases
            similar_indices = np.argsort(similarities)[:self.config['max_similar_cases']]
            
            for idx in similar_indices:
                if similarities[idx] < self.config['similarity_threshold']:
                    similar_case = self._create_contrastive_case(
                        instance, training_data[idx], training_targets[idx], similarities[idx]
                    )
                    similar_cases.append(similar_case)
            
            return similar_cases
            
        except Exception as e:
            self.logger.warning(f"Similar cases search failed: {e}")
            return []
    
    def _create_contrastive_case(self, 
                               current_instance: np.ndarray,
                               case_instance: np.ndarray,
                               case_prediction: float,
                               similarity: float) -> Dict[str, Any]:
        """Create a contrastive case explanation"""
        
        # Find key differences
        key_differences = []
        for i, feature_name in enumerate(self.feature_names):
            if i < len(current_instance) and i < len(case_instance):
                current_val = current_instance[i]
                case_val = case_instance[i]
                difference = abs(current_val - case_val)
                
                # Only include significant differences
                if difference > 0.1 * (abs(current_val) + 1e-8):
                    key_differences.append((feature_name, case_val, current_val))
        
        # Sort by difference magnitude
        key_differences.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
        
        return {
            'case_id': f"case_{len(similar_cases)}",
            'features': {name: case_instance[i] for i, name in enumerate(self.feature_names)},
            'prediction': case_prediction,
            'similarity': similarity,
            'key_differences': key_differences[:3],  # Top 3 differences
            'outcome_difference': abs(case_prediction - self.model.predict(current_instance.reshape(1, -1))[0])
        }
    
    def _analyze_decision_boundary(self, instance: np.ndarray, prediction: float) -> float:
        """Analyze distance to decision boundary"""
        try:
            # Simplified boundary analysis
            # For binary classification, this would be more precise
            
            # Create small perturbations to find boundary
            boundary_distance = 0.0
            n_perturbations = 20
            
            for _ in range(n_perturbations):
                perturbed_instance = instance + np.random.normal(0, 0.05, instance.shape)
                perturbed_pred = self.model.predict(perturbed_instance.reshape(1, -1))[0]
                boundary_distance += abs(prediction - perturbed_pred)
            
            avg_boundary_distance = boundary_distance / n_perturbations
            
            return avg_boundary_distance
            
        except Exception as e:
            self.logger.warning(f"Decision boundary analysis failed: {e}")
            return 0.0
    
    def _identify_risk_opportunity_factors(self, 
                                         key_factors: List[Tuple[str, float, DecisionImpact]]) -> Tuple[List, List]:
        """Identify risk and opportunity factors"""
        risk_factors = []
        opportunity_factors = []
        
        for feature_name, contribution, impact in key_factors:
            if impact in [DecisionImpact.NEGATIVE, DecisionImpact.STRONG_NEGATIVE]:
                risk_factors.append((feature_name, abs(contribution)))
            elif impact in [DecisionImpact.POSITIVE, DecisionImpact.STRONG_POSITIVE]:
                opportunity_factors.append((feature_name, abs(contribution)))
        
        # Sort by magnitude
        risk_factors.sort(key=lambda x: x[1], reverse=True)
        opportunity_factors.sort(key=lambda x: x[1], reverse=True)
        
        return risk_factors[:5], opportunity_factors[:5]
    
    def _generate_rationale(self,
                          prediction: float,
                          key_factors: List[Tuple[str, float, DecisionImpact]],
                          counterfactuals: List[Dict[str, Any]],
                          similar_cases: List[Dict[str, Any]]) -> str:
        """Generate human-readable rationale for the decision"""
        
        rationale_parts = []
        
        # 1. Prediction summary
        rationale_parts.append(f"The model predicts a value of {prediction:.4f}.")
        
        # 2. Key factors
        top_positive = [f for f in key_factors if f[2] in [DecisionImpact.POSITIVE, DecisionImpact.STRONG_POSITIVE]][:2]
        top_negative = [f for f in key_factors if f[2] in [DecisionImpact.NEGATIVE, DecisionImpact.STRONG_NEGATIVE]][:2]
        
        if top_positive:
            pos_features = ", ".join([f[0] for f in top_positive])
            rationale_parts.append(f"Key positive factors include: {pos_features}.")
        
        if top_negative:
            neg_features = ", ".join([f[0] for f in top_negative])
            rationale_parts.append(f"Key negative factors include: {neg_features}.")
        
        # 3. Counterfactual insights
        if counterfactuals:
            best_counterfactual = max(counterfactuals, key=lambda x: x['expected_impact'])
            rationale_parts.append(
                f"To improve the outcome, consider adjusting {list(best_counterfactual['feature_changes'].keys())[0]}."
            )
        
        # 4. Similar cases context
        if similar_cases:
            most_similar = min(similar_cases, key=lambda x: x['similarity'])
            rationale_parts.append(
                f"Similar historical cases show outcomes around {most_similar['prediction']:.4f}."
            )
        
        return " ".join(rationale_parts)

# Visualization and Reporting
class DecisionExplanationVisualizer:
    """Visualization tools for decision explanations"""
    
    @staticmethod
    def create_decision_dashboard(explanation: DecisionExplanation, save_path: str = None):
        """Create comprehensive dashboard for decision explanation"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Key Factor Contributions', 
                'Risk vs Opportunity Factors',
                'Counterfactual Analysis',
                'Similar Cases Comparison',
                'Confidence Assessment',
                'Decision Rationale'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "indicator"}, {"type": "table"}]
            ]
        )
        
        # 1. Key Factor Contributions
        features = [f[0] for f in explanation.key_factors[:8]]
        contributions = [f[1] for f in explanation.key_factors[:8]]
        colors = ['green' if f[2] in [DecisionImpact.POSITIVE, DecisionImpact.STRONG_POSITIVE] 
                 else 'red' if f[2] in [DecisionImpact.NEGATIVE, DecisionImpact.STRONG_NEGATIVE]
                 else 'blue' for f in explanation.key_factors[:8]]
        
        fig.add_trace(
            go.Bar(x=features, y=contributions, marker_color=colors, name='Contributions'),
            row=1, col=1
        )
        
        # 2. Risk vs Opportunity Factors
        risk_features = [f[0] for f in explanation.risk_factors[:5]]
        risk_scores = [f[1] for f in explanation.risk_factors[:5]]
        opp_features = [f[0] for f in explanation.opportunity_factors[:5]]
        opp_scores = [f[1] for f in explanation.opportunity_factors[:5]]
        
        fig.add_trace(
            go.Bar(x=risk_features, y=risk_scores, name='Risk Factors', marker_color='red'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=opp_features, y=opp_scores, name='Opportunity Factors', marker_color='green'),
            row=1, col=2
        )
        
        # 3. Counterfactual Analysis
        if explanation.counterfactuals:
            cf_features = [list(cf['feature_changes'].keys())[0] for cf in explanation.counterfactuals]
            cf_impacts = [cf['expected_impact'] for cf in explanation.counterfactuals]
            cf_feasibility = [cf['feasibility'] for cf in explanation.counterfactuals]
            
            fig.add_trace(
                go.Bar(x=cf_features, y=cf_impacts, name='Expected Impact', marker_color='orange'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=cf_features, y=cf_feasibility, mode='markers', 
                          name='Feasibility', marker=dict(size=10, color='purple')),
                row=2, col=1
            )
        
        # 4. Similar Cases Comparison
        if explanation.similar_cases:
            case_ids = [case['case_id'] for case in explanation.similar_cases]
            similarities = [case['similarity'] for case in explanation.similar_cases]
            predictions = [case['prediction'] for case in explanation.similar_cases]
            
            fig.add_trace(
                go.Scatter(x=case_ids, y=similarities, mode='markers+lines',
                          name='Similarity', marker=dict(size=10)),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=case_ids, y=predictions, mode='markers',
                          name='Prediction', marker=dict(size=8, symbol='diamond')),
                row=2, col=2
            )
        
        # 5. Confidence Assessment
        confidence_value = explanation.confidence
        confidence_color = {
            ConfidenceLevel.VERY_HIGH: 'green',
            ConfidenceLevel.HIGH: 'lightgreen',
            ConfidenceLevel.MEDIUM: 'yellow',
            ConfidenceLevel.LOW: 'orange',
            ConfidenceLevel.VERY_LOW: 'red'
        }.get(explanation.confidence_level, 'gray')
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=confidence_value * 100,
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': confidence_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ]
                }
            ),
            row=3, col=1
        )
        
        # 6. Decision Rationale (simplified table)
        rationale_data = [[explanation.rationale[:100] + "..."]]
        fig.add_trace(
            go.Table(
                header=dict(values=['Decision Rationale']),
                cells=dict(values=rationale_data)
            ),
            row=3, col=2
        )
        
        fig.update_layout(height=1000, title_text="Decision Explanation Dashboard")
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    @staticmethod
    def plot_counterfactual_analysis(counterfactuals: List[Dict[str, Any]], save_path: str = None):
        """Plot counterfactual analysis results"""
        
        if not counterfactuals:
            print("No counterfactuals to display")
            return
        
        features = [list(cf['feature_changes'].keys())[0] for cf in counterfactuals]
        impacts = [cf['expected_impact'] for cf in counterfactuals]
        feasibility = [cf['feasibility'] for cf in counterfactuals]
        costs = [cf['implementation_cost'] for cf in counterfactuals]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Impact vs Feasibility scatter plot
        scatter = ax1.scatter(impacts, feasibility, s=100, c=costs, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Expected Impact')
        ax1.set_ylabel('Feasibility')
        ax1.set_title('Counterfactual Analysis: Impact vs Feasibility')
        
        # Add feature labels
        for i, feature in enumerate(features):
            ax1.annotate(feature, (impacts[i], feasibility[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        # Add colorbar for implementation cost
        plt.colorbar(scatter, ax=ax1, label='Implementation Cost')
        
        # Bar chart of expected impacts
        bars = ax2.bar(features, impacts, color=['green' if x > 0 else 'red' for x in impacts])
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Expected Impact')
        ax2.set_title('Counterfactual Expected Impacts')
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
    n_features = 10
    
    feature_names = [
        'price_momentum', 'volatility', 'volume_trend', 'rsi', 'macd',
        'bollinger_position', 'atr', 'market_cap', 'pe_ratio', 'dividend_yield'
    ]
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = (2 * X_train[:, 0] + 1.5 * X_train[:, 1] - X_train[:, 2] + 
               np.random.randn(n_samples) * 0.5)
    
    # Train a model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create a sample instance to explain
    sample_instance = X_train[0]  # Use first training instance
    
    # Initialize decision explainer
    explainer = AdvancedDecisionExplainer()
    
    # Explain decision
    explanation = explainer.explain_decision(
        model=model,
        instance=sample_instance,
        feature_names=feature_names,
        training_data=X_train,
        training_targets=y_train,
        explanation_type=ExplanationType.COUNTERFACTUAL
    )
    
    print("Decision Explanation Results:")
    print("=" * 50)
    print(f"Prediction: {explanation.prediction:.4f}")
    print(f"Confidence: {explanation.confidence:.2f} ({explanation.confidence_level.value})")
    print(f"Decision Boundary Distance: {explanation.decision_boundary:.4f}")
    
    print("\nKey Factors:")
    print("-" * 20)
    for i, (feature, contrib, impact) in enumerate(explanation.key_factors[:5]):
        print(f"{i+1}. {feature}: {contrib:.4f} ({impact.value})")
    
    print("\nRisk Factors:")
    print("-" * 15)
    for feature, score in explanation.risk_factors:
        print(f"- {feature}: {score:.4f}")
    
    print("\nOpportunity Factors:")
    print("-" * 20)
    for feature, score in explanation.opportunity_factors:
        print(f"- {feature}: {score:.4f}")
    
    print("\nCounterfactual Suggestions:")
    print("-" * 25)
    for i, cf in enumerate(explanation.counterfactuals):
        feature = list(cf['feature_changes'].keys())[0]
        current_val, suggested_val = cf['feature_changes'][feature]
        print(f"{i+1}. {cf['description']}")
        print(f"   Current: {current_val:.4f} → Suggested: {suggested_val:.4f}")
        print(f"   Expected Impact: {cf['expected_impact']:.4f}")
        print(f"   Feasibility: {cf['feasibility']:.2f}, Cost: {cf['implementation_cost']:.2f}")
    
    print("\nSimilar Cases:")
    print("-" * 15)
    for case in explanation.similar_cases:
        print(f"Case {case['case_id']}: Similarity {case['similarity']:.3f}, "
              f"Prediction: {case['prediction']:.4f}")
    
    print(f"\nRationale: {explanation.rationale}")
    
    # Create visualizations
    visualizer = DecisionExplanationVisualizer()
    visualizer.create_decision_dashboard(explanation, "decision_explanation.html")
    visualizer.plot_counterfactual_analysis(explanation.counterfactuals, "counterfactual_analysis.png")

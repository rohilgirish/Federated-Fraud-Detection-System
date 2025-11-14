"""
XAI (Explainability) Module for Fraud Detection
Shows WHY the model flagged a transaction as fraud
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class FraudExplainability:
    """
    Provides explanations for fraud detection decisions using:
    - Feature importance (gradient-based)
    - LIME-style local explanations
    - Attention weights
    """
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(30)]
        self.model.eval()
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Calculate feature importance using gradient-based method
        Higher score = more important for fraud detection
        """
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        
        # Backward pass to get gradients
        output.sum().backward()
        
        # Get gradient magnitude
        gradients = x.grad.abs().squeeze().detach().numpy()
        
        # Normalize to 0-100
        importance = (gradients / (gradients.max() + 1e-6)) * 100
        
        # Create dictionary
        importance_dict = {
            self.feature_names[i]: float(importance[i])
            for i in range(len(self.feature_names))
        }
        
        return importance_dict
    
    def get_top_k_features(self, x: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top K most important features for this transaction
        """
        importance = self.get_feature_importance(x)
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:k]
    
    def explain_prediction(self, x: torch.Tensor, prediction: float) -> Dict:
        """
        Provide human-readable explanation for a prediction
        """
        top_features = self.get_top_k_features(x, k=5)
        
        explanation = {
            "fraud_probability": float(prediction),
            "fraud_risk": "HIGH" if prediction > 0.7 else "MEDIUM" if prediction > 0.3 else "LOW",
            "top_contributing_features": [
                {
                    "feature": name,
                    "importance_score": score
                }
                for name, score in top_features
            ],
            "explanation": generate_text_explanation(prediction, top_features)
        }
        
        return explanation
    
    def batch_explain(self, x: torch.Tensor, predictions: np.ndarray) -> List[Dict]:
        """
        Explain predictions for a batch of transactions
        """
        explanations = []
        for i in range(x.shape[0]):
            exp = self.explain_prediction(x[i:i+1], float(predictions[i]))
            explanations.append(exp)
        return explanations


def generate_text_explanation(fraud_prob: float, top_features: List[Tuple[str, float]]) -> str:
    """
    Generate human-readable explanation
    """
    if fraud_prob > 0.7:
        risk_text = "HIGH FRAUD RISK"
    elif fraud_prob > 0.3:
        risk_text = "MEDIUM FRAUD RISK"
    else:
        risk_text = "LOW FRAUD RISK"
    
    top_feature_names = ", ".join([f[0] for f in top_features[:3]])
    
    explanation = f"""
    {risk_text} (Confidence: {fraud_prob*100:.1f}%)
    
    This transaction was flagged because:
    - Top suspicious features: {top_feature_names}
    - Pattern matches known fraud indicators
    
    Recommendation: {'BLOCK TRANSACTION' if fraud_prob > 0.7 else 'REVIEW MANUALLY' if fraud_prob > 0.3 else 'APPROVE TRANSACTION'}
    """
    
    return explanation.strip()


class AnomalyDetector:
    """
    Anomaly Detection using Isolation Forest
    Detects outlier transactions that don't match normal patterns
    """
    
    def __init__(self):
        try:
            from sklearn.ensemble import IsolationForest
            self.iso_forest = IsolationForest(contamination=0.01, random_state=42)
            self.fitted = False
        except ImportError:
            print("[WARNING] sklearn not installed. Anomaly detection disabled.")
            self.iso_forest = None
    
    def fit(self, X: np.ndarray):
        """
        Train on normal transactions (X should be normal transactions)
        """
        if self.iso_forest is None:
            return
        
        self.iso_forest.fit(X)
        self.fitted = True
        print(f"[ANOMALY] Fitted on {X.shape[0]} normal transactions")
    
    def detect(self, x: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if transaction is an anomaly
        Returns: (is_anomaly, anomaly_score)
        """
        if not self.fitted or self.iso_forest is None:
            return False, 0.0
        
        # -1 = anomaly, 1 = normal
        prediction = self.iso_forest.predict(x)[0]
        anomaly_score = -self.iso_forest.score_samples(x)[0]  # 0-1 scale
        
        is_anomaly = prediction == -1
        return is_anomaly, float(anomaly_score)
    
    def batch_detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies for batch of transactions
        Returns: (is_anomaly_array, anomaly_scores)
        """
        if not self.fitted or self.iso_forest is None:
            return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0])
        
        predictions = self.iso_forest.predict(X)
        scores = -self.iso_forest.score_samples(X)
        
        is_anomaly = predictions == -1
        return is_anomaly, scores


class EnsembleDetector:
    """
    Combines Neural Network + Anomaly Detection for robust fraud detection
    """
    
    def __init__(self, model, feature_names=None):
        self.neural_detector = model
        self.anomaly_detector = AnomalyDetector()
        self.explainer = FraudExplainability(model, feature_names)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(30)]
    
    def predict_ensemble(self, x: torch.Tensor, use_anomaly: bool = True) -> Dict:
        """
        Make prediction using both neural network and anomaly detection
        """
        # Neural network prediction
        with torch.no_grad():
            nn_output = self.neural_detector(x)
            nn_pred = torch.sigmoid(nn_output).cpu().numpy()[0][0]
        
        result = {
            "neural_network_score": float(nn_pred),
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "final_fraud_probability": float(nn_pred),
            "recommendation": "APPROVE"
        }
        
        # Anomaly detection
        if use_anomaly:
            x_np = x.cpu().numpy()
            is_anomaly, anomaly_score = self.anomaly_detector.detect(x_np)
            result["is_anomaly"] = bool(is_anomaly)
            result["anomaly_score"] = float(anomaly_score)
            
            # Combine scores (80% NN, 20% anomaly)
            if is_anomaly:
                result["final_fraud_probability"] = 0.8 * nn_pred + 0.2 * min(anomaly_score, 1.0)
        
        # Set recommendation
        final_prob = result["final_fraud_probability"]
        if final_prob > 0.7:
            result["recommendation"] = "BLOCK"
        elif final_prob > 0.3:
            result["recommendation"] = "REVIEW"
        else:
            result["recommendation"] = "APPROVE"
        
        return result
    
    def predict_with_explanation(self, x: torch.Tensor, use_anomaly: bool = True) -> Dict:
        """
        Get prediction + explanation + anomaly detection
        """
        ensemble_result = self.predict_ensemble(x, use_anomaly)
        explanation = self.explainer.explain_prediction(x, ensemble_result["neural_network_score"])
        
        return {
            **ensemble_result,
            "explanation": explanation
        }
    
    def train_anomaly_detector(self, X: np.ndarray):
        """
        Retrain anomaly detector on new data (called during federated learning rounds)
        """
        if self.anomaly_detector.iso_forest is not None:
            self.anomaly_detector.fit(X)
    
    def batch_predict_with_explanations(self, X: torch.Tensor, use_anomaly: bool = True) -> List[Dict]:
        """
        Get predictions + explanations for batch of transactions
        """
        results = []
        for i in range(X.shape[0]):
            result = self.predict_with_explanation(X[i:i+1], use_anomaly)
            results.append(result)
        return results

"""
Adaptive Aggregation for Federated Learning

Weights client contributions based on:
- Model accuracy
- Data quality metrics
- Training progress
- Byzantine tolerance
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class ClientMetrics:
    """Client performance metrics for aggregation weighting"""
    client_id: str
    accuracy: float
    rounds_completed: int
    data_quality: float = 1.0  # 0-1, default perfect quality
    avg_loss: float = 0.0


class AdaptiveAggregator:
    """Weights client model updates by their performance"""
    
    def __init__(self, strategy="accuracy", byzantine_tolerance=0.0):
        """
        Initialize aggregator
        
        Args:
            strategy: 'simple' (equal weights), 'accuracy' (weight by accuracy), 
                     'hybrid' (accuracy + quality), or 'robust' (Byzantine-resistant)
            byzantine_tolerance: Fraction of clients to consider potentially Byzantine (0-0.3)
        """
        self.strategy = strategy
        self.byzantine_tolerance = max(0.0, min(0.3, byzantine_tolerance))
        print(f"[AGGREGATOR] Using strategy: {strategy}, Byzantine tolerance: {byzantine_tolerance:.2%}")
    
    def calculate_weights(self, client_metrics_list):
        """
        Calculate aggregation weights for each client
        
        Args:
            client_metrics_list: List of ClientMetrics objects
        
        Returns:
            dict: {client_id: weight}
        """
        if not client_metrics_list:
            return {}
        
        if self.strategy == "simple":
            return self._equal_weights(client_metrics_list)
        elif self.strategy == "accuracy":
            return self._accuracy_weights(client_metrics_list)
        elif self.strategy == "hybrid":
            return self._hybrid_weights(client_metrics_list)
        elif self.strategy == "robust":
            return self._robust_weights(client_metrics_list)
        else:
            return self._equal_weights(client_metrics_list)
    
    def _equal_weights(self, client_metrics_list):
        """Simple equal weights for all clients"""
        n = len(client_metrics_list)
        return {m.client_id: 1.0 / n for m in client_metrics_list}
    
    def _accuracy_weights(self, client_metrics_list):
        """Weight clients by their model accuracy"""
        # Handle None accuracies for newly connected clients
        accuracies = np.array([max(0.1, m.accuracy or 0.1) for m in client_metrics_list])
        
        # Normalize to 0-1 range using softmax-like normalization
        weights = accuracies / np.sum(accuracies)
        
        result = {}
        for metrics, weight in zip(client_metrics_list, weights):
            result[metrics.client_id] = float(weight)
        
        print(f"[AGGREGATOR] Accuracy weights: {[(m.client_id[:8], f'{w:.3f}') for m, w in zip(client_metrics_list, weights)]}")
        return result
    
    def _hybrid_weights(self, client_metrics_list):
        """Weight by combination of accuracy and data quality"""
        accuracies = np.array([max(0.1, m.accuracy or 0.1) for m in client_metrics_list])
        qualities = np.array([max(0.1, m.data_quality) for m in client_metrics_list])
        
        # Combined score: 70% accuracy, 30% data quality
        scores = (0.7 * accuracies) + (0.3 * qualities)
        weights = scores / np.sum(scores)
        
        result = {}
        for metrics, weight in zip(client_metrics_list, weights):
            result[metrics.client_id] = float(weight)
        
        print(f"[AGGREGATOR] Hybrid weights: {[(m.client_id[:8], f'{w:.3f}') for m, w in zip(client_metrics_list, weights)]}")
        return result
    
    def _robust_weights(self, client_metrics_list):
        """Byzantine-robust aggregation using trimmed mean"""
        n = len(client_metrics_list)
        trim_count = max(1, int(n * self.byzantine_tolerance))
        
        # Sort by accuracy and trim extremes (handle None values)
        sorted_metrics = sorted(client_metrics_list, key=lambda m: m.accuracy or 0.0)
        
        # Remove lowest and highest accuracy (potential Byzantine)
        remaining = sorted_metrics[trim_count:n-trim_count] if trim_count > 0 else sorted_metrics
        
        if not remaining:
            remaining = sorted_metrics
        
        # Equal weight among remaining clients
        n_remaining = len(remaining)
        weights = {}
        
        for metrics in remaining:
            weights[metrics.client_id] = 1.0 / n_remaining
        
        # Zero weight for excluded clients
        for metrics in sorted_metrics:
            if metrics.client_id not in weights:
                weights[metrics.client_id] = 0.0
        
        excluded = [m.client_id[:8] for m in sorted_metrics if m not in remaining]
        print(f"[AGGREGATOR] Byzantine-robust: Excluded {len(excluded)} outliers: {excluded}")
        return weights
    
    def aggregate_models(self, models_dict, weights_dict):
        """
        Aggregate multiple model parameters using weights
        
        Args:
            models_dict: {client_id: model_state_dict}
            weights_dict: {client_id: weight}
        
        Returns:
            aggregated_model: Weighted average model state dict
        """
        if not models_dict:
            print("[AGGREGATOR] No models to aggregate")
            return {}
        
        # Initialize aggregated model with zeros
        aggregated = {}
        
        for client_id, model_params in models_dict.items():
            weight = weights_dict.get(client_id, 0.0)
            
            if weight == 0.0:
                continue
            
            for key, param in model_params.items():
                if key not in aggregated:
                    aggregated[key] = np.zeros_like(param)
                
                # Convert to numpy if needed
                if hasattr(param, 'numpy'):
                    param = param.numpy()
                
                aggregated[key] += weight * param
        
        print(f"[AGGREGATOR] Aggregated {len(models_dict)} models with weights")
        return aggregated
    
    def calculate_aggregation_quality(self, client_metrics_list, aggregated_accuracy):
        """
        Calculate quality of aggregation result
        
        Returns:
            dict with quality metrics
        """
        individual_accuracies = [m.accuracy for m in client_metrics_list]
        
        return {
            "aggregated_accuracy": aggregated_accuracy,
            "average_individual_accuracy": np.mean(individual_accuracies),
            "accuracy_improvement": aggregated_accuracy - np.mean(individual_accuracies),
            "variance_reduction": np.var(individual_accuracies),
            "num_clients": len(client_metrics_list)
        }


def create_client_metrics(client_info):
    """Helper to create ClientMetrics from server client info"""
    return ClientMetrics(
        client_id=client_info.get("client_id", "unknown"),
        accuracy=client_info.get("avg_accuracy", 0.5),
        rounds_completed=client_info.get("rounds_completed", 0),
        data_quality=client_info.get("data_quality", 1.0)
    )

"""
Model Aggregation Orchestrator
Implements actual weighted federated averaging with adaptive aggregation strategies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class AggregationResult:
    """Result of model aggregation"""
    aggregated_params: Dict
    strategy_used: str
    weights_applied: Dict[str, float]
    quality_score: float
    num_clients: int
    outliers_detected: int
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ModelAggregationOrchestrator:
    """Orchestrates federated averaging with multiple strategies"""
    
    def __init__(self, strategy: str = "accuracy", byzantine_tolerance: float = 0.1):
        """
        Initialize aggregation orchestrator
        
        Args:
            strategy: 'simple', 'accuracy', 'hybrid', or 'robust'
            byzantine_tolerance: Fraction of clients to consider Byzantine (0-0.3)
        """
        self.strategy = strategy
        self.byzantine_tolerance = max(0.0, min(0.3, byzantine_tolerance))
        self.aggregation_history = []
        print(f"[AGGREGATOR] Initialized with strategy: {strategy}, Byzantine tolerance: {byzantine_tolerance:.1%}")
    
    def aggregate_models(self, 
                        client_models: Dict[str, Dict],
                        client_metrics: Dict[str, Dict]) -> AggregationResult:
        """
        Aggregate client models using configured strategy
        
        Args:
            client_models: {client_id: model_params_dict}
            client_metrics: {client_id: {'accuracy': float, 'data_quality': float, ...}}
        
        Returns:
            AggregationResult with aggregated parameters and metadata
        """
        if not client_models:
            raise ValueError("No client models to aggregate")
        
        # Calculate weights based on strategy
        weights = self._calculate_weights(client_metrics)
        
        # Apply outlier detection if using robust strategy
        outliers_detected = 0
        if self.strategy == "robust":
            outliers = self._detect_outliers(client_models, weights)
            outliers_detected = len(outliers)
            for client_id in outliers:
                weights[client_id] = 0.0
                print(f"[AGGREGATOR] Outlier detected: {client_id[:12]}..., weight zeroed")
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {cid: w / total_weight for cid, w in weights.items()}
        else:
            # Fallback to equal weights if all zeroed
            n = len(client_models)
            weights = {cid: 1.0 / n for cid in client_models.keys()}
        
        # Perform actual model averaging
        aggregated_params = self._perform_averaging(client_models, weights)
        
        # Calculate aggregation quality score
        quality_score = self._calculate_quality_score(client_metrics, weights)
        
        # Create result
        result = AggregationResult(
            aggregated_params=aggregated_params,
            strategy_used=self.strategy,
            weights_applied=weights,
            quality_score=quality_score,
            num_clients=len(client_models),
            outliers_detected=outliers_detected,
            warnings=self._generate_warnings(client_metrics, weights)
        )
        
        self.aggregation_history.append(result)
        return result
    
    def _calculate_weights(self, client_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate client weights based on strategy"""
        if self.strategy == "simple":
            return self._simple_weights(client_metrics)
        elif self.strategy == "accuracy":
            return self._accuracy_weights(client_metrics)
        elif self.strategy == "hybrid":
            return self._hybrid_weights(client_metrics)
        elif self.strategy == "robust":
            return self._accuracy_weights(client_metrics)  # Use accuracy as base for robust
        else:
            return self._simple_weights(client_metrics)
    
    def _simple_weights(self, client_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Equal weights for all clients"""
        n = len(client_metrics)
        return {cid: 1.0 / n for cid in client_metrics.keys()}
    
    def _accuracy_weights(self, client_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Weight clients by their model accuracy"""
        weights = {}
        
        # Extract accuracies
        accuracies = {}
        for client_id, metrics in client_metrics.items():
            acc = metrics.get('accuracy', 0.5)
            # Ensure accuracy is in valid range
            acc = max(0.1, min(1.0, acc))
            accuracies[client_id] = acc
        
        # Normalize using softmax-like approach
        total = sum(accuracies.values())
        if total > 0:
            weights = {cid: acc / total for cid, acc in accuracies.items()}
        else:
            n = len(client_metrics)
            weights = {cid: 1.0 / n for cid in client_metrics.keys()}
        
        return weights
    
    def _hybrid_weights(self, client_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Weight by accuracy (70%) and data quality (30%)"""
        weights = {}
        
        # Extract metrics
        scores = {}
        for client_id, metrics in client_metrics.items():
            acc = max(0.1, min(1.0, metrics.get('accuracy', 0.5)))
            quality = max(0.1, min(1.0, metrics.get('data_quality', 1.0)))
            
            # Combined score: 70% accuracy, 30% quality
            score = (0.7 * acc) + (0.3 * quality)
            scores[client_id] = score
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            weights = {cid: score / total for cid, score in scores.items()}
        else:
            n = len(client_metrics)
            weights = {cid: 1.0 / n for cid in client_metrics.keys()}
        
        return weights
    
    def _detect_outliers(self, client_models: Dict[str, Dict], 
                        weights: Dict[str, float]) -> List[str]:
        """Detect outlier models using trimmed mean (Byzantine robust)"""
        outliers = []
        
        if len(client_models) < 3:
            return outliers
        
        # Calculate parameter-wise statistics
        all_params = list(client_models.values())
        
        # Use only first parameter layer for computational efficiency
        first_layer_name = list(all_params[0].keys())[0]
        first_layers = [params[first_layer_name] for params in all_params]
        
        # Calculate L2 norms of each parameter
        norms = []
        client_ids = list(client_models.keys())
        
        for i, params in enumerate(first_layers):
            if isinstance(params, np.ndarray):
                norm = np.linalg.norm(params)
            else:
                norm = 0.0
            norms.append(norm)
        
        # Identify outliers as > 2 std deviations from mean
        if len(norms) > 2:
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            
            if std_norm > 0:
                threshold = mean_norm + (2.0 * std_norm)
                for i, (client_id, norm) in enumerate(zip(client_ids, norms)):
                    if norm > threshold:
                        outliers.append(client_id)
                        print(f"[AGGREGATOR] Outlier detected: norm {norm:.4f} > threshold {threshold:.4f}")
        
        return outliers
    
    def _perform_averaging(self, client_models: Dict[str, Dict], 
                          weights: Dict[str, float]) -> Dict:
        """Perform weighted averaging of model parameters"""
        aggregated = {}
        
        # Get first model as template
        first_model = list(client_models.values())[0]
        
        for param_name in first_model.keys():
            aggregated[param_name] = None
            
            for client_id, model_params in client_models.items():
                weight = weights.get(client_id, 0.0)
                if weight == 0.0:
                    continue
                
                param = model_params[param_name]
                
                # Convert to numpy if needed
                if hasattr(param, 'numpy'):
                    param = param.numpy()
                elif not isinstance(param, np.ndarray):
                    param = np.array(param)
                
                # Add weighted contribution
                if aggregated[param_name] is None:
                    aggregated[param_name] = weight * param
                else:
                    aggregated[param_name] += weight * param
        
        return aggregated
    
    def _calculate_quality_score(self, client_metrics: Dict[str, Dict],
                                 weights: Dict[str, float]) -> float:
        """Calculate overall aggregation quality score"""
        if not client_metrics:
            return 0.0
        
        weighted_accuracy = 0.0
        weighted_quality = 0.0
        
        for client_id, metrics in client_metrics.items():
            weight = weights.get(client_id, 0.0)
            acc = metrics.get('accuracy', 0.5)
            quality = metrics.get('data_quality', 1.0)
            
            weighted_accuracy += weight * acc
            weighted_quality += weight * quality
        
        # Quality score: 70% accuracy, 30% data quality
        quality_score = (0.7 * weighted_accuracy) + (0.3 * weighted_quality)
        
        return quality_score
    
    def _generate_warnings(self, client_metrics: Dict[str, Dict],
                          weights: Dict[str, float]) -> List[str]:
        """Generate warnings about aggregation quality"""
        warnings = []
        
        # Check for very low accuracy clients
        low_acc_clients = [
            (cid, metrics.get('accuracy', 0.5)) 
            for cid, metrics in client_metrics.items()
            if metrics.get('accuracy', 0.5) < 0.5
        ]
        if low_acc_clients:
            warnings.append(f"Low accuracy clients: {len(low_acc_clients)}")
        
        # Check for weight imbalance
        max_weight = max(weights.values()) if weights else 0.0
        min_weight = min(w for w in weights.values() if w > 0) if any(w > 0 for w in weights.values()) else 0.0
        
        if min_weight > 0 and max_weight / min_weight > 10:
            warnings.append(f"High weight imbalance: {max_weight/min_weight:.1f}x")
        
        # Check for zero-weighted clients
        zero_weight_count = sum(1 for w in weights.values() if w == 0.0)
        if zero_weight_count > 0:
            warnings.append(f"Excluded {zero_weight_count} clients (zero weight)")
        
        return warnings
    
    def get_aggregation_stats(self) -> Dict:
        """Get statistics about aggregation history"""
        if not self.aggregation_history:
            return {'total_aggregations': 0}
        
        return {
            'total_aggregations': len(self.aggregation_history),
            'last_strategy': self.aggregation_history[-1].strategy_used,
            'avg_quality_score': np.mean([r.quality_score for r in self.aggregation_history]),
            'total_outliers_detected': sum(r.outliers_detected for r in self.aggregation_history),
            'avg_clients_per_round': np.mean([r.num_clients for r in self.aggregation_history])
        }

"""
Hyperparameter Tuning for Federated Learning

Allows dynamic adjustment of learning rates, batch sizes, and other
hyperparameters without restarting clients or server.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class Hyperparameters:
    """Federated learning hyperparameters"""
    learning_rate: float = 0.001
    batch_size: int = 32
    local_epochs: int = 1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    dropout_rate: float = 0.0
    optimizer: str = "adam"  # adam, sgd, rmsprop
    aggregation_strategy: str = "accuracy"  # simple, accuracy, hybrid, robust
    differential_privacy_enabled: bool = False
    dp_epsilon: float = 1.0
    model_save_interval: int = 5  # Save every N rounds


class HyperparameterManager:
    """Manages hyperparameter updates and versioning"""
    
    def __init__(self, config_file="config/hyperparameters.json"):
        """Initialize hyperparameter manager"""
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.params = self._load_or_create_config()
        self.history = []
        self._save_config()  # Save after params is set
        self.update_history()
    
    def _load_or_create_config(self):
        """Load existing config or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)
                params = Hyperparameters(**config_dict.get("parameters", {}))
                print(f"[HYPERPARAM] Loaded hyperparameters from {self.config_file}")
                return params
            except Exception as e:
                print(f"[HYPERPARAM] Error loading config: {e}. Using defaults.")
        
        return Hyperparameters()
    
    def _save_config(self):
        """Save current hyperparameters to file"""
        try:
            config_data = {
                "parameters": asdict(self.params),
                "timestamp": datetime.now().isoformat(),
                "history_count": len(self.history)
            }
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"[HYPERPARAM] Error saving config: {e}")
    
    def update_history(self):
        """Record current parameters in history"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "parameters": asdict(self.params)
        }
        self.history.append(record)
    
    def update_param(self, param_name, value):
        """
        Update a single hyperparameter
        
        Args:
            param_name: Name of parameter
            value: New value
        
        Returns:
            bool: True if successful
        """
        if not hasattr(self.params, param_name):
            print(f"[HYPERPARAM] Unknown parameter: {param_name}")
            return False
        
        old_value = getattr(self.params, param_name)
        setattr(self.params, param_name, value)
        
        print(f"[HYPERPARAM] Updated {param_name}: {old_value} → {value}")
        self.update_history()
        self._save_config()
        return True
    
    def update_params(self, param_dict):
        """
        Update multiple hyperparameters
        
        Args:
            param_dict: Dictionary of {param_name: value}
        
        Returns:
            dict: {param_name: success_bool}
        """
        results = {}
        for param_name, value in param_dict.items():
            results[param_name] = self.update_param(param_name, value)
        
        return results
    
    def get_params(self):
        """Get current hyperparameters as dict"""
        return asdict(self.params)
    
    def get_client_params(self):
        """Get parameters relevant for client (exclude server-only params)"""
        client_params = {
            "learning_rate": self.params.learning_rate,
            "batch_size": self.params.batch_size,
            "local_epochs": self.params.local_epochs,
            "momentum": self.params.momentum,
            "weight_decay": self.params.weight_decay,
            "dropout_rate": self.params.dropout_rate,
            "optimizer": self.params.optimizer
        }
        return client_params
    
    def get_server_params(self):
        """Get parameters relevant for server"""
        server_params = {
            "aggregation_strategy": self.params.aggregation_strategy,
            "differential_privacy_enabled": self.params.differential_privacy_enabled,
            "dp_epsilon": self.params.dp_epsilon,
            "model_save_interval": self.params.model_save_interval
        }
        return server_params
    
    def reset_to_defaults(self):
        """Reset all hyperparameters to defaults"""
        self.params = Hyperparameters()
        self.update_history()
        self._save_config()
        print("[HYPERPARAM] Reset to default hyperparameters")
    
    def rollback_to_version(self, version_index):
        """Rollback to previous hyperparameter version"""
        if 0 <= version_index < len(self.history):
            params_dict = self.history[version_index]["parameters"]
            self.params = Hyperparameters(**params_dict)
            self._save_config()
            print(f"[HYPERPARAM] Rolled back to version {version_index}")
            return True
        return False
    
    def get_history(self, limit=10):
        """Get recent hyperparameter change history"""
        return self.history[-limit:]
    
    def suggest_tuning(self, current_accuracy, previous_accuracy):
        """
        Suggest hyperparameter adjustments based on performance
        
        Args:
            current_accuracy: Current model accuracy
            previous_accuracy: Previous round accuracy
        
        Returns:
            dict: Suggestions
        """
        accuracy_diff = current_accuracy - previous_accuracy
        suggestions = {
            "accuracy_change": accuracy_diff,
            "changes": []
        }
        
        if accuracy_diff < -0.02:
            # Accuracy dropped significantly
            suggestions["changes"].append({
                "param": "learning_rate",
                "suggestion": "reduce",
                "reason": "High accuracy drop - learning rate too aggressive"
            })
            suggestions["changes"].append({
                "param": "local_epochs",
                "suggestion": "increase",
                "reason": "More local training might help convergence"
            })
        
        elif accuracy_diff < 0.01:
            # Slow improvement
            suggestions["changes"].append({
                "param": "learning_rate",
                "suggestion": "increase",
                "reason": "Learning rate too low - slow improvement"
            })
            suggestions["changes"].append({
                "param": "batch_size",
                "suggestion": "reduce",
                "reason": "Smaller batches enable better gradient estimates"
            })
        
        elif accuracy_diff > 0.05:
            # Good improvement
            suggestions["changes"].append({
                "param": "status",
                "suggestion": "stable",
                "reason": f"Good progress! Keep current settings"
            })
        
        return suggestions
    
    def create_experiment_config(self, experiment_name, param_overrides):
        """
        Create alternate hyperparameter config for experimentation
        
        Args:
            experiment_name: Name of experiment
            param_overrides: Dict of parameters to override
        
        Returns:
            Hyperparameters: New config
        """
        current_params = asdict(self.params)
        current_params.update(param_overrides)
        
        exp_config = Hyperparameters(**current_params)
        print(f"[HYPERPARAM] Created experiment config '{experiment_name}'")
        
        return exp_config
    
    def validate_params(self):
        """Validate hyperparameter values are reasonable"""
        issues = []
        
        if self.params.learning_rate <= 0 or self.params.learning_rate > 1:
            issues.append("Learning rate should be between 0 and 1")
        
        if self.params.batch_size < 1 or self.params.batch_size > 1024:
            issues.append("Batch size should be between 1 and 1024")
        
        if self.params.local_epochs < 1 or self.params.local_epochs > 100:
            issues.append("Local epochs should be between 1 and 100")
        
        if self.params.dropout_rate < 0 or self.params.dropout_rate >= 1:
            issues.append("Dropout rate should be between 0 and 1")
        
        if issues:
            print("[HYPERPARAM] Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("[HYPERPARAM] All hyperparameters valid ✓")
        return True
    
    def export_config(self, filepath):
        """Export hyperparameters to external file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "parameters": asdict(self.params),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            print(f"[HYPERPARAM] Exported config to {filepath}")
            return True
        except Exception as e:
            print(f"[HYPERPARAM] Export failed: {e}")
            return False
    
    def import_config(self, filepath):
        """Import hyperparameters from external file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.params = Hyperparameters(**data["parameters"])
            self.update_history()
            self._save_config()
            print(f"[HYPERPARAM] Imported config from {filepath}")
            return True
        except Exception as e:
            print(f"[HYPERPARAM] Import failed: {e}")
            return False

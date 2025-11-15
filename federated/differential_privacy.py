"""
Differential Privacy for Federated Learning

Implements DP-SGD with gradient clipping and Laplace/Gaussian noise
for privacy-preserving federated training.
"""
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DifferentialPrivacyConfig:
    """Configuration for differential privacy"""
    epsilon: float = 1.0  # Privacy budget (lower = more private)
    delta: float = 1e-5   # Probability of privacy breach
    clipping_norm: float = 1.0  # Gradient clipping threshold
    noise_multiplier: float = 1.0  # Multiplier for noise scale
    enable: bool = True
    noise_type: str = 'laplace'  # 'laplace' or 'gaussian'


class DifferentialPrivacyMechanism:
    """Implements differential privacy for model updates with DP-SGD"""
    
    def __init__(self, config=None):
        """Initialize with DP config"""
        self.config = config or DifferentialPrivacyConfig()
        self.gradient_history = []
        self.privacy_spent = 0.0
        self.total_noise_added = 0.0
        print(f"[DP] Differential Privacy enabled: {self.config.enable}")
        if self.config.enable:
            print(f"[DP] Privacy budget (ε): {self.config.epsilon}")
            print(f"[DP] Failure probability (δ): {self.config.delta}")
            print(f"[DP] Gradient clipping norm: {self.config.clipping_norm}")
            print(f"[DP] Noise type: {self.config.noise_type}")
            print(f"[DP] Gradient clipping norm: {self.config.clipping_norm}")
    
    def clip_gradients(self, model_params):
        """
        Clip gradients to a maximum norm for sensitivity limiting
        
        Args:
            model_params: Model parameters (dict or array-like)
        
        Returns:
            clipped_params: Gradients clipped to max norm
        """
        if isinstance(model_params, dict):
            clipped = {}
            norm_sum = 0.0
            
            # Calculate total norm
            for key, param in model_params.items():
                if hasattr(param, 'numpy'):
                    param = param.numpy()
                norm_sum += np.sum(param ** 2)
            
            total_norm = np.sqrt(norm_sum)
            
            # Clip if necessary
            clipping_factor = min(1.0, self.config.clipping_norm / (total_norm + 1e-10))
            
            for key, param in model_params.items():
                if hasattr(param, 'numpy'):
                    param = param.numpy()
                clipped[key] = param * clipping_factor
            
            if clipping_factor < 1.0:
                print(f"[DP] Clipped gradients (factor: {clipping_factor:.3f})")
            
            return clipped
        else:
            # Handle numpy array case
            norm = np.linalg.norm(model_params)
            clipping_factor = min(1.0, self.config.clipping_norm / (norm + 1e-10))
            return model_params * clipping_factor
    
    def add_noise(self, model_params):
        """
        Add Laplace noise for differential privacy (using Laplace mechanism)
        
        Args:
            model_params: Model parameters to noise
        
        Returns:
            noisy_params: Parameters with added noise
        """
        if not self.config.enable:
            return model_params
        
        # Calculate noise scale using Laplace mechanism
        # For DP-SGD: sigma = sqrt(2 * ln(1.25/delta)) / epsilon * clipping_norm
        sensitivity = self.config.clipping_norm
        
        sigma = (np.sqrt(2 * np.log(1.25 / self.config.delta)) / self.config.epsilon) * sensitivity
        sigma *= self.config.noise_multiplier
        
        noisy_params = {}
        
        if isinstance(model_params, dict):
            for key, param in model_params.items():
                if hasattr(param, 'numpy'):
                    param = param.numpy()
                
                # Generate Laplace noise with same shape
                noise = np.random.laplace(0, sigma, size=param.shape)
                noisy_params[key] = param + noise
            
            print(f"[DP] Added Laplace noise (σ={sigma:.6f}) for privacy")
            return noisy_params
        else:
            noise = np.random.laplace(0, sigma, size=model_params.shape)
            return model_params + noise
    
    def add_gaussian_noise(self, model_params, num_clients=1):
        """
        Add Gaussian noise using DP-SGD mechanism
        
        Args:
            model_params: Model parameters
            num_clients: Number of clients for scaling
        
        Returns:
            noisy_params: Parameters with Gaussian noise
        """
        if not self.config.enable:
            return model_params
        
        # DP-SGD noise scale
        sensitivity = self.config.clipping_norm
        
        # Accounts mechanism noise scaling
        sigma = (sensitivity * self.config.noise_multiplier * np.sqrt(2 * np.log(1.25 / self.config.delta))) / self.config.epsilon
        sigma /= np.sqrt(num_clients)  # Scale down with aggregation
        
        noisy_params = {}
        
        if isinstance(model_params, dict):
            for key, param in model_params.items():
                if hasattr(param, 'numpy'):
                    param = param.numpy()
                
                # Generate Gaussian noise
                noise = np.random.normal(0, sigma, size=param.shape)
                noisy_params[key] = param + noise
            
            print(f"[DP] Added Gaussian noise (σ={sigma:.6f}) for aggregation privacy")
            return noisy_params
        else:
            noise = np.random.normal(0, sigma, size=model_params.shape)
            return model_params + noise
    
    def privatize_update(self, model_params, strategy="laplace"):
        """
        Complete DP pipeline: clip then add noise
        
        Args:
            model_params: Model parameters
            strategy: 'laplace' or 'gaussian'
        
        Returns:
            privatized_params: Clipped and noised parameters
        """
        if not self.config.enable:
            return model_params
        
        # Step 1: Clip gradients
        clipped = self.clip_gradients(model_params)
        
        # Step 2: Add noise
        if strategy == "gaussian":
            noisy = self.add_gaussian_noise(clipped)
        else:  # laplace
            noisy = self.add_noise(clipped)
        
        return noisy
    
    def compute_epsilon_spent(self, num_updates, composition="basic"):
        """
        Estimate total privacy budget spent
        
        Args:
            num_updates: Number of model updates/rounds
            composition: 'basic', 'advanced', or 'renyi'
        
        Returns:
            dict: Total epsilon and delta spent
        """
        if composition == "basic":
            # Basic composition: epsilon multiplied by number of rounds
            total_epsilon = self.config.epsilon * num_updates
        elif composition == "advanced":
            # Advanced composition: sqrt(2*log(1.5/delta)*num_updates)*epsilon + epsilon*log(1.5/delta)
            term1 = np.sqrt(2 * np.log(1.5 / self.config.delta) * num_updates) * self.config.epsilon
            term2 = self.config.epsilon * np.log(1.5 / self.config.delta)
            total_epsilon = term1 + term2
        else:  # renyi
            # Rényi composition (best case)
            lambda_param = np.ceil(np.log(num_updates) / np.log(2))
            total_epsilon = self.config.epsilon * lambda_param * np.log(2)
        
        return {
            "epsilon_spent": total_epsilon,
            "delta": self.config.delta,
            "num_rounds": num_updates,
            "composition_type": composition,
            "privacy_level": "Strong ✓" if total_epsilon < 1.0 else ("Moderate ✓" if total_epsilon < 5.0 else "Weak")
        }
    
    def update_config(self, epsilon=None, delta=None, clipping_norm=None, noise_multiplier=None):
        """Update DP configuration dynamically"""
        if epsilon is not None:
            self.config.epsilon = epsilon
        if delta is not None:
            self.config.delta = delta
        if clipping_norm is not None:
            self.config.clipping_norm = clipping_norm
        if noise_multiplier is not None:
            self.config.noise_multiplier = noise_multiplier
        
        print(f"[DP] Updated config: ε={self.config.epsilon}, δ={self.config.delta}")


def create_dp_config(privacy_level="moderate"):
    """
    Create DP config preset by privacy level
    
    Args:
        privacy_level: 'low' (more private), 'moderate', 'high' (less private)
    
    Returns:
        DifferentialPrivacyConfig
    """
    configs = {
        "low": DifferentialPrivacyConfig(epsilon=0.5, delta=1e-6, noise_multiplier=2.0),
        "moderate": DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, noise_multiplier=1.0),
        "high": DifferentialPrivacyConfig(epsilon=5.0, delta=1e-4, noise_multiplier=0.5)
    }
    return configs.get(privacy_level, configs["moderate"])

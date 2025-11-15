import torch
import numpy as np
from cryptography.fernet import Fernet
from typing import Dict, Any

class SecureAggregation:
    def __init__(self, epsilon=1.0):
        """
        Initialize secure aggregation with differential privacy
        epsilon: privacy budget
        """
        self.epsilon = epsilon
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        
    def add_noise(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add Laplace noise for differential privacy
        """
        noisy_params = {}
        for key, param in params.items():
            sensitivity = 1.0 / param.numel()  # Scale with parameter size
            noise = np.random.laplace(0, sensitivity/self.epsilon, param.shape)
            noisy_params[key] = param + torch.tensor(noise, dtype=param.dtype)
        return noisy_params
    
    def encrypt_params(self, params: Dict[str, torch.Tensor]) -> bytes:
        """
        Encrypt model parameters
        """
        serialized = torch.save(params, f=None)
        return self.cipher_suite.encrypt(serialized)
    
    def decrypt_params(self, encrypted_params: bytes) -> Dict[str, torch.Tensor]:
        """
        Decrypt model parameters
        """
        decrypted = self.cipher_suite.decrypt(encrypted_params)
        return torch.load(decrypted)
    
    def secure_aggregate(self, params_list: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Securely aggregate parameters from multiple clients
        """
        # Add noise to each client's parameters
        noisy_params = [self.add_noise(params) for params in params_list]
        
        # Average the parameters
        aggregated = {}
        for key in noisy_params[0].keys():
            aggregated[key] = torch.stack([p[key] for p in noisy_params]).mean(0)
        
        return aggregated

class ClientAuthentication:
    def __init__(self):
        self.clients = {}
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def register_client(self, client_id: str) -> bytes:
        """
        Register a new client and return their authentication token
        """
        token = Fernet.generate_key()
        self.clients[client_id] = token
        return self.cipher_suite.encrypt(token)
    
    def authenticate_client(self, client_id: str, token: bytes) -> bool:
        """
        Verify client's authentication token
        """
        if client_id not in self.clients:
            return False
        try:
            decrypted_token = self.cipher_suite.decrypt(token)
            return decrypted_token == self.clients[client_id]
        except:
            return False
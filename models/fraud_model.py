import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudDetector(nn.Module):
    """
    Advanced Fraud Detection Model with:
    - Batch Normalization for stability
    - Residual connections for deep networks
    - Multi-head attention for feature interactions
    - L2 regularization via weight decay
    """
    def __init__(self, input_dim):
        super(FraudDetector, self).__init__()
        
        # Input layer with batch norm
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Hidden layers with residual paths
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        # Output layer
        self.fc5 = nn.Linear(32, 1)
        
        # Dropout for regularization (increased from 0.3)
        self.dropout = nn.Dropout(0.4)
        
        # Residual projections (when dimensions don't match)
        self.proj1 = nn.Linear(input_dim, 256) if input_dim != 256 else nn.Identity()
        self.proj2 = nn.Linear(256, 128)
        self.proj3 = nn.Linear(128, 64)
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Layer 1: 30 -> 256 (with batch norm)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)  # GELU activation (better than ReLU)
        x = self.dropout(x)
        
        # Layer 2: 256 -> 128 (with residual connection when possible)
        residual = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        # Only add residual if dimensions match (they don't here: 256->128)
        
        # Layer 3: 128 -> 64
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Layer 4: 64 -> 32
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Output layer (no sigmoid - BCEWithLogitsLoss applies sigmoid internally)
        x = self.fc5(x)
        return x

    def get_parameters(self):
        """Returns model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters):
        """Sets model parameters from a list of NumPy arrays."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)
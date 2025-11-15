import torch
import os
import sys

# Ensure repo root import
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from models.fraud_model import FraudDetector

if __name__ == '__main__':
    model = FraudDetector(input_dim=30)
    save_path = os.path.join(root, 'models', 'fraud_detection_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved new fraud_detection_model.pth to {save_path}")

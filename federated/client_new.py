import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import socket
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fraud_model import FraudDetector
import time

class FederatedClient:
    def __init__(self, host='localhost', port=8085):
        self.host = host
        self.port = port
        self.socket = None
        self.max_retries = 3
        
    def connect(self):
        for _ in range(self.max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self.socket.connect((self.host, self.port))
                print("[CLIENT] Connected to server")
                return True
            except Exception as e:
                print(f"[CLIENT] Connection attempt failed: {e}")
                if self.socket:
                    self.socket.close()
                    self.socket = None
                time.sleep(2)  # Wait before retrying
        return False
        
    def recv_msg(self):
        try:
            size_data = self.socket.recv(4)
            if not size_data:
                return None
            size = int.from_bytes(size_data, 'big')
            
            data = bytearray()
            while len(data) < size:
                chunk = self.socket.recv(min(size - len(data), 8192))
                if not chunk:
                    return None
                data.extend(chunk)
            return data
        except Exception as e:
            print(f"[CLIENT] Error receiving message: {e}")
            return None
    
    def send_msg(self, data):
        try:
            size = len(data)
            self.socket.send(size.to_bytes(4, 'big'))
            self.socket.sendall(data)
            return True
        except Exception as e:
            print(f"[CLIENT] Error sending message: {e}")
            return False

def load_data(file_path):
    print("[CLIENT] Loading data...")
    df = pd.read_csv(file_path)
    X = torch.tensor(df.drop('Class', axis=1).values, dtype=torch.float32)
    y = torch.tensor(df['Class'].values, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
    return DataLoader(train, batch_size=32), DataLoader(test, batch_size=32)

def train(model, loader):
    print("[CLIENT] Training model...")
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for X, y in loader:
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds.squeeze(), y)
        loss.backward()
        optimizer.step()
    return model.state_dict()

def test(model, loader):
    print("[CLIENT] Testing model...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            preds = model(X)
            correct += ((preds.squeeze() > 0.5) == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    print("[CLIENT] Starting client...")
    client = FederatedClient()
    if not client.connect():
        print("[CLIENT] Failed to connect to server")
        return

    try:
        # Load data and model
        trainloader, testloader = load_data("data/kaggle_fraud.csv")
        model = FraudDetector(input_dim=30)
        
        # Training rounds
        for round in range(3):
            print(f"\n[CLIENT] Starting round {round + 1}")
            
            # Train the model
            model_params = train(model, trainloader)
            accuracy = test(model, testloader)
            
            # Example metrics
            message = {
                'type': 'model_update',
                'params': model_params,
                'accuracy': accuracy,
                'fairness_score': 0.85,
                'fairness': 0.80
            }
            
            # Send update to server
            print(f"[CLIENT] Sending round {round + 1} update to server")
            client.send_msg(pickle.dumps(message))
            
            # Get response from server
            data = client.recv_msg()
            if data:
                response = pickle.loads(data)
                if response['type'] == 'model_update':
                    model.load_state_dict(response['params'])
                    print(f"[CLIENT] Received updated model from server")
            
            print(f"[CLIENT] Round {round + 1} completed")
            print(f"Accuracy: {accuracy*100:.2f}%")
            time.sleep(1)  # Pause between rounds
            
    except Exception as e:
        print(f"[CLIENT] Error during training: {e}")
    finally:
        if client.socket:
            client.socket.close()

if __name__ == "__main__":
    main()
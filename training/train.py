import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FraudDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        return torch.sigmoid(x)

def preprocess_data():
    # Load the data
    data = pd.read_csv('data/kaggle_fraud.csv')
    
    # Separate features and labels
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(scaler, 'models/scaler.pth')
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

def train_model(X_train, X_test, y_train, y_test, batch_size=64, epochs=10, learning_rate=0.001):
    # Create data loaders
    train_dataset = FraudDataset(X_train, y_train)
    test_dataset = FraudDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize the model
    input_dim = X_train.shape[1]
    model = FraudDetectionModel(input_dim)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.view(-1, 1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    outputs = model(batch_features)
                    predicted = (outputs >= 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted.view(-1) == batch_labels).sum().item()
            
            accuracy = correct / total
            print(f'Validation Accuracy: {accuracy:.4f}')
    
    # Save the trained model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/fraud_detection_model.pth')
    return model

def main():
    print("Starting data preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data()
    
    print("Starting model training...")
    model = train_model(X_train, X_test, y_train, y_test)
    print("Training completed! Model saved in models/fraud_detection_model.pth")

if __name__ == "__main__":
    main()
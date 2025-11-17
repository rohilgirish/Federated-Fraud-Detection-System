# FairFinance: Enterprise Federated Fraud Detection Framework

## Project Summary

FairFinance is a production-grade, privacy-preserving federated learning (FL) system specifically engineered for the financial services sector. The framework allows decentralized banking institutions to collectively train an advanced fraud detection model on their local datasets without ever transferring raw transaction data across the network. By leveraging differential privacy, secure model management, and explainable AI, FairFinance provides a secure alternative to risky centralized data pooling.

## The Problem: Data Silos vs. Fraud Sophistication

Fraudulent activities are increasingly global and multi-institutional. Traditional fraud detection models suffer from "data silo" syndrome, where a bank can only train models on its own restricted data. This results in models that are blind to emerging fraud patterns observed elsewhere in the industry.

FairFinance solves this by enabling collaborative intelligence while strictly adhering to data sovereignty regulations such as GDPR, CCPA, and HIPAA.

## Core Pillars

### 1. Privacy-Preserving Architecture
The system utilizes Local Differential Privacy (LDP). Before a client sends model weights to the central server, the framework applies gradient clipping and mathematical noise (Laplace or Gaussian mechanisms). This ensures that the individual contribution of any specific client or transaction cannot be reverse-engineered by an adversary, even one with access to the server.

### 2. Byzantine-Robust Aggregation
Real-world federated learning can be targeted by "poisoning" attacks where a malicious client sends bad updates to degrade the global model. FairFinance implements an Adaptive Aggregation Orchestrator with several strategies:
*   **Accuracy-Weighted Aggregation:** Prioritizes updates from clients performing better on validation sets.
*   **Robust Hybrid Strategy:** Automatically detects and filters out outlier updates that deviate significantly from the population mean.

### 3. Integrated Explainable AI (XAI)
Transparency is mandatory in financial decision-making. Our framework uses an ensemble approach:
*   **Neural Network Attributions:** Calculates gradient-based feature importance to show which fields (Amount, Location, Frequency) influenced a "High Risk" classification.
*   **Anomaly Detection:** A secondary Isolation Forest model runs in parallel to provide a "Fairness Check," ensuring that the model isn't just memorizing patterns but identifying actual anomalies.

## Technical Architecture and Directory Structure

The project is organized into a modular hierarchy to support professional deployment:

### Operations and Logic
*   **`/federated`**: Contains the core FL protocols.
    *   `server.py`: Manages the central loop, versioning, and secure aggregation logic.
    *   `client.py`: Handles local dataset training and privacy-preserving weight submission.
*   **`/models`**: The mathematical core.
    *   `fraud_model.py`: High-performance MLP with residual layers and GELU activations.
    *   `explainability.py`: The suite for gradient-based feature importance and ensemble detection.
    *   `model_compression.py`: Optimized logic for reducing model size for edge deployment.

### System Assets
*   **`/config`**: YAML-based configuration for managing 50+ system parameters (thresholds, learning rates, privacy budgets).
*   **`/data`**: Local transaction storage (CSV/Parquet formats).
*   **`/utils`**: Shared system utilities including centralized logging and metric persistence.
*   **`/scripts`**: Operational automation tools (data generators, system health checks).
*   **`/dashboard`**: Streamlit-based monitoring frontend.
*   **`/logs`**: Persistent audit trails and server state snapshots stored in SQLite databases.

### Outputs
*   **`/training`**: Centralized location for model training scripts and historical performance logs.
    *   `train.py`: Standalone baseline training script for benchmarker comparison.
    *   `training_history.json`: Core data file for dashboard metrics persistence.

## Advanced Features in Detail

### The 'FairFinance' Scoring System
Unlike standard accuracy metrics, we utilize a specialized fairness score. This score penalizes models that have high disparity in fraud detection across different demographic or income groups, ensuring the model remains ethical and unbiased.

### Model Versioning and Rollback
The server maintains a versioned repository of the global model. If a training round is detected to have been compromised (e.g., sudden drop in accuracy or fairness), the supervisor can trigger a rollback to the last known-good state through the dashboard.

### Real-time Hyperparameter Management
Advanced users can tune the following parameters "on-the-fly":
*   **Epsilon (ε) Budget:** Control the strength of the Differential Privacy.
*   **Aggregation Mode:** Switch between Simple, Accuracy-weighted, or Robust.
*   **Training Intensity:** Modify local epochs and batch sizes to balance network traffic vs. convergence speed.

## Deployment Guide

### Environment Setup
The project uses a standard virtual environment approach:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### Running a Federated Round
1.  **Initialize the Server**: `python federated/server.py`
2.  **Launch the Control Panel**: `python -m streamlit run dashboard/app.py`
3.  **Deploy Clients**: `python federated/client.py --name Client_A`

## Data Privacy and Compliance Statement
This software is designed to facilitate compliance with data privacy laws by ensuring that no Personally Identifiable Information (PII) is ever transmitted. The noise addition meets modern differential privacy standards for ε-differential privacy, suitable for production banking environments.

## License
This project is distributed under the MIT License. Full details can be found in the [LICENSE](LICENSE) file.

"""
Metrics persistence layer - saves training history to JSON file
Completely safe - doesn't modify existing code, just adds new functionality
"""

import json
import os
from datetime import datetime
from pathlib import Path

class MetricsHistory:
    """Save and load federated learning metrics to/from JSON"""
    
    def __init__(self, filename='training_history.json'):
        """
        Initialize metrics history handler
        
        Args:
            filename: Name of JSON file to store metrics
        """
        self.filename = filename
        self.history = []
        self.load()
    
    def load(self):
        """Load existing metrics from file"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.history = json.load(f)
                print(f"[METRICS] Loaded {len(self.history)} existing records from {self.filename}")
            else:
                self.history = []
        except Exception as e:
            print(f"[METRICS] Warning: Could not load history: {e}")
            self.history = []
    
    def add_round(self, round_num, accuracy, fairness, fairness_score, 
                  communication, robustness, active_clients, clients_data=None):
        """
        Save metrics for a training round
        
        Args:
            round_num: Round number
            accuracy: Overall accuracy (0-1)
            fairness: Fairness metric (0-1)
            fairness_score: Combined fairness score (0-1)
            communication: Communication efficiency (0-1)
            robustness: System robustness (0-1)
            active_clients: Number of active clients
            clients_data: Optional detailed client data
        """
        entry = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'accuracy': round(accuracy, 4),
            'fairness': round(fairness, 4),
            'fairness_score': round(fairness_score, 4),
            'communication': round(communication, 4),
            'robustness': round(robustness, 4),
            'active_clients': active_clients,
            'clients': clients_data or []
        }
        
        self.history.append(entry)
        self._save()
        
        return entry
    
    def add_client_update(self, round_num, client_id, client_name, accuracy, fairness):
        """
        Save individual client update
        
        Args:
            round_num: Round number
            client_id: Unique client ID
            client_name: Client friendly name
            accuracy: Client accuracy
            fairness: Client fairness metric
        """
        entry = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'client_name': client_name,
            'accuracy': round(accuracy, 4),
            'fairness': round(fairness, 4)
        }
        
        if not hasattr(self, 'client_history'):
            self.client_history = []
        
        self.client_history.append(entry)
        self._save_client_history()
        
        return entry
    
    def _save(self):
        """Save metrics to JSON file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"[METRICS] Error saving metrics: {e}")
    
    def _save_client_history(self):
        """Save client history to separate file"""
        try:
            client_file = self.filename.replace('.json', '_clients.json')
            with open(client_file, 'w') as f:
                json.dump(self.client_history, f, indent=2)
        except Exception as e:
            print(f"[METRICS] Error saving client history: {e}")
    
    def get_latest(self):
        """Get latest round data"""
        return self.history[-1] if self.history else None
    
    def get_round(self, round_num):
        """Get specific round data"""
        for entry in self.history:
            if entry.get('round') == round_num:
                return entry
        return None
    
    def get_improvement(self):
        """Get improvement from first to last round"""
        if len(self.history) < 2:
            return None
        
        first = self.history[0]
        last = self.history[-1]
        
        return {
            'rounds': len(self.history),
            'accuracy_improvement': round(last['accuracy'] - first['accuracy'], 4),
            'fairness_improvement': round(last['fairness'] - first['fairness'], 4),
            'fairness_score_improvement': round(last['fairness_score'] - first['fairness_score'], 4)
        }
    
    def export_csv(self, filename=None):
        """Export metrics to CSV for analysis"""
        if not filename:
            filename = self.filename.replace('.json', '.csv')
        
        try:
            import csv
            if not self.history:
                print("[METRICS] No history to export")
                return
            
            keys = self.history[0].keys()
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
            print(f"[METRICS] Exported to {filename}")
        except Exception as e:
            print(f"[METRICS] Error exporting CSV: {e}")
    
    def clear(self):
        """Clear history (use with caution!)"""
        self.history = []
        self._save()
        print("[METRICS] History cleared")
    
    def summary(self):
        """Print summary of training progress"""
        if not self.history:
            print("[METRICS] No training history")
            return
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total rounds: {len(self.history)}")
        
        latest = self.history[-1]
        print(f"\nLatest Round ({latest['round']}):")
        print(f"  Accuracy:        {latest['accuracy']:.2%}")
        print(f"  Fairness:        {latest['fairness']:.2%}")
        print(f"  Fairness Score:  {latest['fairness_score']:.2%}")
        print(f"  Active Clients:  {latest['active_clients']}")
        
        improvement = self.get_improvement()
        if improvement:
            print(f"\nImprovement (Round 1 → Round {improvement['rounds']}):")
            print(f"  Accuracy:       {improvement['accuracy_improvement']:+.2%}")
            print(f"  Fairness:       {improvement['fairness_improvement']:+.2%}")
            print(f"  Fairness Score: {improvement['fairness_score_improvement']:+.2%}")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Test the metrics saver
    metrics = MetricsHistory()
    
    # Simulate some training rounds
    for round_num in range(1, 6):
        accuracy = 0.7 + (round_num * 0.02)
        fairness = 0.65 + (round_num * 0.03)
        fairness_score = (accuracy + fairness) / 2
        
        metrics.add_round(
            round_num=round_num,
            accuracy=accuracy,
            fairness=fairness,
            fairness_score=fairness_score,
            communication=0.85 + (round_num * 0.01),
            robustness=0.75 + (round_num * 0.01),
            active_clients=3
        )
    
    metrics.summary()
    metrics.export_csv()

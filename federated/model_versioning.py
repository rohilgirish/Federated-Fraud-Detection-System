"""
Model Versioning System for Federated Learning

Tracks model versions, allows rollback to previous versions,
and maintains version history with metadata.
"""
import json
import os
from datetime import datetime
from pathlib import Path


class ModelVersionManager:
    def __init__(self, version_dir="models/versions"):
        """Initialize version manager with storage directory"""
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.version_dir / "versions_metadata.json"
        self.versions = self._load_metadata()
    
    def _load_metadata(self):
        """Load version metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[VERSIONING] Error loading metadata: {e}")
        return {
            "versions": [],
            "current_version": None,
            "best_version": None,
            "best_accuracy": 0.0
        }
    
    def _save_metadata(self):
        """Save version metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            print(f"[VERSIONING] Error saving metadata: {e}")
    
    def save_version(self, model_params, round_num, accuracy, metadata=None):
        """
        Save a new model version
        
        Args:
            model_params: Model state dict to save
            round_num: Current training round
            accuracy: Model accuracy achieved
            metadata: Additional metadata dict
        
        Returns:
            version_id: Unique version identifier
        """
        version_id = f"v{round_num}_{int(datetime.now().timestamp())}"
        version_path = self.version_dir / version_id
        version_path.mkdir(exist_ok=True)
        
        # Save model params
        import pickle
        model_file = version_path / "model_params.pkl"
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model_params, f)
        except Exception as e:
            print(f"[VERSIONING] Error saving model: {e}")
            return None
        
        # Create version record
        version_record = {
            "version_id": version_id,
            "round": round_num,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "status": "saved"
        }
        
        self.versions["versions"].append(version_record)
        self.versions["current_version"] = version_id
        
        # Track best version by accuracy
        if accuracy > self.versions.get("best_accuracy", 0.0):
            self.versions["best_version"] = version_id
            self.versions["best_accuracy"] = accuracy
        
        self._save_metadata()
        print(f"[VERSIONING] Saved version {version_id} with accuracy {accuracy:.4f}")
        
        return version_id
    
    def load_version(self, version_id):
        """Load model params from a specific version"""
        version_path = self.version_dir / version_id
        model_file = version_path / "model_params.pkl"
        
        if not model_file.exists():
            print(f"[VERSIONING] Version {version_id} not found")
            return None
        
        try:
            import pickle
            with open(model_file, 'rb') as f:
                params = pickle.load(f)
            print(f"[VERSIONING] Loaded version {version_id}")
            return params
        except Exception as e:
            print(f"[VERSIONING] Error loading version: {e}")
            return None
    
    def get_current_version(self):
        """Get the current model version"""
        if self.versions["current_version"]:
            return self.load_version(self.versions["current_version"])
        return None
    
    def get_best_version(self):
        """Get the best model version (highest accuracy)"""
        if self.versions["best_version"]:
            return self.load_version(self.versions["best_version"])
        return None
    
    def rollback_to_version(self, version_id):
        """Rollback to a previous model version"""
        params = self.load_version(version_id)
        if params:
            self.versions["current_version"] = version_id
            self._save_metadata()
            print(f"[VERSIONING] Rolled back to {version_id}")
            return params
        return None
    
    def list_versions(self, limit=10):
        """List available versions (latest first)"""
        versions_list = sorted(
            self.versions["versions"],
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
        return versions_list
    
    def get_version_info(self, version_id):
        """Get metadata for a specific version"""
        for v in self.versions["versions"]:
            if v["version_id"] == version_id:
                return v
        return None
    
    def compare_versions(self, version_id_1, version_id_2):
        """Compare two versions"""
        v1 = self.get_version_info(version_id_1)
        v2 = self.get_version_info(version_id_2)
        
        if not v1 or not v2:
            return None
        
        return {
            "version_1": v1,
            "version_2": v2,
            "accuracy_diff": v2["accuracy"] - v1["accuracy"],
            "round_diff": v2["round"] - v1["round"]
        }
    
    def cleanup_old_versions(self, keep_count=20):
        """Remove old versions keeping only the most recent N"""
        versions_list = sorted(
            self.versions["versions"],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        to_remove = versions_list[keep_count:]
        
        import shutil
        for v in to_remove:
            version_path = self.version_dir / v["version_id"]
            try:
                shutil.rmtree(version_path)
                self.versions["versions"].remove(v)
                print(f"[VERSIONING] Removed old version {v['version_id']}")
            except Exception as e:
                print(f"[VERSIONING] Error removing version: {e}")
        
        self._save_metadata()

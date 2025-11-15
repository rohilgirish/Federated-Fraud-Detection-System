"""
Configuration Management System
YAML-based runtime configuration for server and clients
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "localhost"
    port: int = 8085
    heartbeat_timeout: int = 120
    idle_timeout: int = 300
    max_consecutive_timeouts: int = 3
    aggregation_strategy: str = "accuracy"
    byzantine_tolerance: float = 0.1
    enable_differential_privacy: bool = False
    model_save_interval: int = 5
    checkpoint_enabled: bool = True
    persistence_db: str = "logs/server_state.db"
    

@dataclass
class ClientConfig:
    """Client configuration"""
    host: str = "localhost"
    port: int = 8085
    learning_rate: float = 0.001
    batch_size: int = 32
    local_epochs: int = 1
    data_file: str = "data/creditcard.csv"
    max_rounds: int = 10
    connection_retries: int = 3
    connection_retry_delay: int = 2


class ConfigManager:
    """Manages configuration from YAML files with defaults"""
    
    DEFAULT_SERVER_CONFIG = {
        'server': {
            'host': 'localhost',
            'port': 8085,
            'heartbeat_timeout': 120,
            'idle_timeout': 300,
            'max_consecutive_timeouts': 3,
            'aggregation_strategy': 'accuracy',
            'byzantine_tolerance': 0.1,
            'enable_differential_privacy': False,
            'model_save_interval': 5,
            'checkpoint_enabled': True,
            'persistence_db': 'logs/server_state.db'
        },
        'logging': {
            'level': 'INFO',
            'log_dir': 'logs'
        }
    }
    
    DEFAULT_CLIENT_CONFIG = {
        'client': {
            'host': 'localhost',
            'port': 8085,
            'learning_rate': 0.001,
            'batch_size': 32,
            'local_epochs': 1,
            'data_file': 'data/creditcard.csv',
            'max_rounds': 10,
            'connection_retries': 3,
            'connection_retry_delay': 2
        }
    }
    
    def __init__(self, config_type: str = "server", config_file: str = None):
        """
        Initialize config manager
        
        Args:
            config_type: 'server' or 'client'
            config_file: Path to YAML config file (uses default if not provided)
        """
        self.config_type = config_type
        self.config_file = config_file
        self.config = self._load_config()
        print(f"[CONFIG] Loaded {config_type} configuration")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        
        # Start with defaults
        if self.config_type == "server":
            config = self.DEFAULT_SERVER_CONFIG.copy()
            default_file = "config/server_config.yaml"
        else:
            config = self.DEFAULT_CLIENT_CONFIG.copy()
            default_file = "config/client_config.yaml"
        
        # Override with file if provided
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
                print(f"[CONFIG] Loaded from {self.config_file}")
            except Exception as e:
                print(f"[CONFIG] Error loading {self.config_file}: {e}, using defaults")
        elif Path(default_file).exists():
            try:
                with open(default_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
                print(f"[CONFIG] Loaded from {default_file}")
            except Exception as e:
                print(f"[CONFIG] Error loading {default_file}: {e}, using defaults")
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key (dot notation supported)"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set config value by key (dot notation supported)"""
        keys = key.split('.')
        config = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            return True
        except Exception as e:
            print(f"[CONFIG] Error setting {key}: {e}")
            return False
    
    def save(self, filepath: str) -> bool:
        """Save current config to YAML file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"[CONFIG] Saved to {filepath}")
            return True
        except Exception as e:
            print(f"[CONFIG] Error saving config: {e}")
            return False
    
    def get_server_config(self) -> ServerConfig:
        """Get server config as dataclass"""
        server_cfg = self.config.get('server', {})
        return ServerConfig(
            host=server_cfg.get('host', 'localhost'),
            port=server_cfg.get('port', 8085),
            heartbeat_timeout=server_cfg.get('heartbeat_timeout', 120),
            idle_timeout=server_cfg.get('idle_timeout', 300),
            max_consecutive_timeouts=server_cfg.get('max_consecutive_timeouts', 3),
            aggregation_strategy=server_cfg.get('aggregation_strategy', 'accuracy'),
            byzantine_tolerance=server_cfg.get('byzantine_tolerance', 0.1),
            enable_differential_privacy=server_cfg.get('enable_differential_privacy', False),
            model_save_interval=server_cfg.get('model_save_interval', 5),
            checkpoint_enabled=server_cfg.get('checkpoint_enabled', True),
            persistence_db=server_cfg.get('persistence_db', 'logs/server_state.db')
        )
    
    def get_client_config(self) -> ClientConfig:
        """Get client config as dataclass"""
        client_cfg = self.config.get('client', {})
        return ClientConfig(
            host=client_cfg.get('host', 'localhost'),
            port=client_cfg.get('port', 8085),
            learning_rate=client_cfg.get('learning_rate', 0.001),
            batch_size=client_cfg.get('batch_size', 32),
            local_epochs=client_cfg.get('local_epochs', 1),
            data_file=client_cfg.get('data_file', 'data/creditcard.csv'),
            max_rounds=client_cfg.get('max_rounds', 10),
            connection_retries=client_cfg.get('connection_retries', 3),
            connection_retry_delay=client_cfg.get('connection_retry_delay', 2)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary"""
        return self.config.copy()
    
    def to_json(self) -> str:
        """Export config as JSON string"""
        return json.dumps(self.config, indent=2)
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print(f"CONFIGURATION SUMMARY ({self.config_type.upper()})")
        print("="*60)
        
        def print_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print(" " * indent + f"{key}:")
                    print_dict(value, indent + 2)
                else:
                    print(" " * indent + f"{key}: {value}")
        
        print_dict(self.config)
        print("="*60 + "\n")


def create_default_configs():
    """Create default configuration files if they don't exist"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Server config
    server_config_file = config_dir / "server_config.yaml"
    if not server_config_file.exists():
        with open(server_config_file, 'w') as f:
            yaml.dump(ConfigManager.DEFAULT_SERVER_CONFIG, f, default_flow_style=False)
        print(f"[CONFIG] Created {server_config_file}")
    
    # Client config
    client_config_file = config_dir / "client_config.yaml"
    if not client_config_file.exists():
        with open(client_config_file, 'w') as f:
            yaml.dump(ConfigManager.DEFAULT_CLIENT_CONFIG, f, default_flow_style=False)
        print(f"[CONFIG] Created {client_config_file}")

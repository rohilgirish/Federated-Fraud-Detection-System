"""
Production Server Manager with Error Handling, Persistence, and Checkpointing
Provides robust server foundation for federated learning
"""

import sqlite3
import json
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import hashlib


@dataclass
class ServerCheckpoint:
    """Model checkpoint with metadata"""
    round_num: int
    model_params: dict
    aggregation_strategy: str
    active_clients: int
    global_accuracy: float
    timestamp: str
    checksum: str = ""
    
    def calculate_checksum(self):
        """Calculate checksum for integrity verification"""
        data_str = str(self.round_num) + str(self.timestamp) + str(self.active_clients)
        self.checksum = hashlib.md5(data_str.encode()).hexdigest()[:12]


class ServerPersistenceManager:
    """Handles database logging, checkpoints, and recovery"""
    
    def __init__(self, db_file='logs/server_state.db', checkpoint_dir='models/checkpoints'):
        """Initialize persistence layer"""
        self.db_file = db_file
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.lock = Lock()
        self._init_database()
        print(f"[PERSISTENCE] Initialized with DB: {db_file}, Checkpoints: {checkpoint_dir}")
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        with self.lock:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Audit trail table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    client_id TEXT,
                    client_name TEXT,
                    round_num INTEGER,
                    message TEXT,
                    severity TEXT DEFAULT 'INFO'
                )
            ''')
            
            # Server state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS server_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_num INTEGER UNIQUE,
                    timestamp TEXT NOT NULL,
                    model_checksum TEXT,
                    global_accuracy REAL,
                    aggregation_strategy TEXT,
                    active_clients INTEGER,
                    status TEXT DEFAULT 'completed'
                )
            ''')
            
            # Client tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS client_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT UNIQUE NOT NULL,
                    client_name TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    rounds_completed INTEGER DEFAULT 0,
                    total_updates INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'connected'
                )
            ''')
            
            # Model aggregation history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aggregation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_num INTEGER,
                    timestamp TEXT,
                    strategy TEXT,
                    client_count INTEGER,
                    weights TEXT,
                    quality_score REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            print("[PERSISTENCE] Database schema initialized")
    
    def log_event(self, event_type: str, message: str, 
                  client_id: str = None, client_name: str = None, 
                  round_num: int = None, severity: str = 'INFO'):
        """Log audit trail event"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_log 
                    (timestamp, event_type, client_id, client_name, round_num, message, severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), event_type, client_id, client_name, 
                      round_num, message, severity))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[PERSISTENCE] Error logging event: {e}")
    
    def log_client_update(self, client_id: str, client_name: str, round_num: int):
        """Track client update"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                
                # Check if client exists
                cursor.execute('SELECT id FROM client_tracking WHERE client_id = ?', (client_id,))
                exists = cursor.fetchone()
                
                now = datetime.now().isoformat()
                
                if exists:
                    cursor.execute('''
                        UPDATE client_tracking 
                        SET last_seen = ?, rounds_completed = ?, total_updates = total_updates + 1
                        WHERE client_id = ?
                    ''', (now, round_num, client_id))
                else:
                    cursor.execute('''
                        INSERT INTO client_tracking 
                        (client_id, client_name, first_seen, last_seen, rounds_completed, total_updates)
                        VALUES (?, ?, ?, ?, ?, 1)
                    ''', (client_id, client_name, now, now, round_num))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[PERSISTENCE] Error tracking client: {e}")
    
    def save_checkpoint(self, checkpoint: ServerCheckpoint):
        """Save model checkpoint with metadata"""
        try:
            checkpoint.calculate_checksum()
            
            # Save to database
            with self.lock:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO server_state 
                    (round_num, timestamp, model_checksum, global_accuracy, aggregation_strategy, active_clients)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (checkpoint.round_num, checkpoint.timestamp, checkpoint.checksum, 
                      checkpoint.global_accuracy, checkpoint.aggregation_strategy, 
                      checkpoint.active_clients))
                conn.commit()
                conn.close()
            
            # Save model file
            checkpoint_file = self.checkpoint_dir / f"checkpoint_r{checkpoint.round_num}_{checkpoint.checksum}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'params': checkpoint.model_params,
                    'metadata': {
                        'round': checkpoint.round_num,
                        'accuracy': checkpoint.global_accuracy,
                        'strategy': checkpoint.aggregation_strategy,
                        'timestamp': checkpoint.timestamp
                    }
                }, f)
            
            print(f"[PERSISTENCE] Saved checkpoint R{checkpoint.round_num}: {checkpoint.checksum}")
            self.log_event('CHECKPOINT_SAVED', f"Round {checkpoint.round_num}, Accuracy: {checkpoint.global_accuracy:.4f}",
                          round_num=checkpoint.round_num, severity='INFO')
            
            return True
        except Exception as e:
            print(f"[PERSISTENCE] Error saving checkpoint: {e}")
            self.log_event('CHECKPOINT_ERROR', str(e), severity='ERROR')
            return False
    
    def load_latest_checkpoint(self) -> Optional[ServerCheckpoint]:
        """Load most recent checkpoint"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT round_num, timestamp, model_checksum, global_accuracy, aggregation_strategy, active_clients
                    FROM server_state
                    ORDER BY round_num DESC
                    LIMIT 1
                ''')
                result = cursor.fetchone()
                conn.close()
            
            if not result:
                print("[PERSISTENCE] No checkpoint found")
                return None
            
            round_num, timestamp, checksum, accuracy, strategy, clients = result
            
            # Find checkpoint file
            checkpoint_file = self.checkpoint_dir / f"checkpoint_r{round_num}_{checksum}.pkl"
            if not checkpoint_file.exists():
                print(f"[PERSISTENCE] Checkpoint file not found: {checkpoint_file}")
                return None
            
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            checkpoint = ServerCheckpoint(
                round_num=round_num,
                model_params=data['params'],
                aggregation_strategy=strategy,
                active_clients=clients,
                global_accuracy=accuracy,
                timestamp=timestamp,
                checksum=checksum
            )
            
            print(f"[PERSISTENCE] Loaded checkpoint from round {round_num}")
            return checkpoint
        except Exception as e:
            print(f"[PERSISTENCE] Error loading checkpoint: {e}")
            return None
    
    def get_client_status_report(self) -> Dict:
        """Get status of all tracked clients"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT client_id, client_name, first_seen, last_seen, rounds_completed, 
                           total_updates, status
                    FROM client_tracking
                    ORDER BY last_seen DESC
                ''')
                results = cursor.fetchall()
                conn.close()
            
            report = {
                'total_clients': len(results),
                'active_clients': 0,
                'inactive_clients': 0,
                'clients': []
            }
            
            now = datetime.now()
            for row in results:
                client_id, name, first_seen, last_seen, rounds, updates, status = row
                last_seen_time = datetime.fromisoformat(last_seen)
                seconds_ago = (now - last_seen_time).total_seconds()
                is_active = seconds_ago < 300  # Active if seen in last 5 minutes
                
                report['clients'].append({
                    'client_id': client_id[:12] + '...',
                    'client_name': name,
                    'rounds_completed': rounds,
                    'total_updates': updates,
                    'last_seen_seconds_ago': seconds_ago,
                    'status': 'Active' if is_active else 'Inactive'
                })
                
                if is_active:
                    report['active_clients'] += 1
                else:
                    report['inactive_clients'] += 1
            
            return report
        except Exception as e:
            print(f"[PERSISTENCE] Error generating report: {e}")
            return {'total_clients': 0, 'active_clients': 0, 'clients': []}
    
    def log_aggregation(self, round_num: int, strategy: str, client_count: int, 
                       weights: Dict[str, float], quality_score: float):
        """Log aggregation strategy execution"""
        try:
            weights_json = json.dumps({k: float(v) for k, v in weights.items()})
            
            with self.lock:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO aggregation_history
                    (round_num, timestamp, strategy, client_count, weights, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (round_num, datetime.now().isoformat(), strategy, client_count, 
                      weights_json, quality_score))
                conn.commit()
                conn.close()
            
            self.log_event('AGGREGATION', f"Strategy: {strategy}, Clients: {client_count}",
                          round_num=round_num)
        except Exception as e:
            print(f"[PERSISTENCE] Error logging aggregation: {e}")
    
    def get_server_health(self) -> Dict:
        """Get comprehensive server health status"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                
                # Get latest round
                cursor.execute('SELECT MAX(round_num), timestamp FROM server_state')
                latest = cursor.fetchone()
                
                # Count events by type
                cursor.execute('''
                    SELECT event_type, COUNT(*) as count FROM audit_log 
                    GROUP BY event_type
                ''')
                events = dict(cursor.fetchall())
                
                conn.close()
            
            last_round = latest[0] if latest[0] else 0
            last_round_time = latest[1] if latest[1] else 'Never'
            
            return {
                'last_round': last_round,
                'last_round_time': last_round_time,
                'total_events': sum(events.values()),
                'event_breakdown': events,
                'db_file': self.db_file,
                'checkpoint_dir': str(self.checkpoint_dir),
                'checkpoint_count': len(list(self.checkpoint_dir.glob('*.pkl')))
            }
        except Exception as e:
            print(f"[PERSISTENCE] Error getting health: {e}")
            return {}

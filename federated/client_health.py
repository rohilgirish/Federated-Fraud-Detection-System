"""
Client Manager with Heartbeat and Timeout Detection
Maintains connection health and detects dead/stale clients
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class ClientStatus(Enum):
    """Client connection status"""
    CONNECTING = "connecting"
    ACTIVE = "active"
    IDLE = "idle"
    STALE = "stale"
    DISCONNECTED = "disconnected"
    TIMEOUT = "timeout"


@dataclass
class ClientHeartbeat:
    """Client heartbeat data"""
    client_id: str
    client_name: str
    last_heartbeat: datetime
    last_update: datetime
    status: ClientStatus
    rounds_completed: int = 0
    consecutive_timeouts: int = 0
    data_quality_score: float = 1.0
    model_accuracy: float = 0.5
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'last_update': self.last_update.isoformat(),
            'status': self.status.value,
            'rounds_completed': self.rounds_completed,
            'consecutive_timeouts': self.consecutive_timeouts,
            'data_quality_score': self.data_quality_score,
            'model_accuracy': self.model_accuracy
        }


class ClientHealthManager:
    """Manages client connection health and timeout detection"""
    
    def __init__(self, heartbeat_timeout: int = 120, idle_timeout: int = 300):
        """
        Initialize client health manager
        
        Args:
            heartbeat_timeout: Seconds before marking client stale (default 2 min)
            idle_timeout: Seconds before marking client disconnected (default 5 min)
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.idle_timeout = idle_timeout
        self.clients: Dict[str, ClientHeartbeat] = {}
        self.max_consecutive_timeouts = 3
        
        print(f"[CLIENT_HEALTH] Initialized with heartbeat timeout: {heartbeat_timeout}s, idle timeout: {idle_timeout}s")
    
    def register_client(self, client_id: str, client_name: str) -> ClientHeartbeat:
        """Register new client"""
        now = datetime.now()
        heartbeat = ClientHeartbeat(
            client_id=client_id,
            client_name=client_name,
            last_heartbeat=now,
            last_update=now,
            status=ClientStatus.CONNECTING,
            rounds_completed=0,
            consecutive_timeouts=0
        )
        self.clients[client_id] = heartbeat
        print(f"[CLIENT_HEALTH] Registered client: {client_name} ({client_id[:12]}...)")
        return heartbeat
    
    def update_heartbeat(self, client_id: str) -> bool:
        """Update client heartbeat (ping received)"""
        if client_id not in self.clients:
            return False
        
        self.clients[client_id].last_heartbeat = datetime.now()
        if self.clients[client_id].status == ClientStatus.STALE:
            self.clients[client_id].consecutive_timeouts = 0
        
        self.clients[client_id].status = ClientStatus.ACTIVE
        return True
    
    def record_update(self, client_id: str, accuracy: float = None, 
                     quality_score: float = None) -> bool:
        """Record model update from client"""
        if client_id not in self.clients:
            return False
        
        client = self.clients[client_id]
        client.last_update = datetime.now()
        client.last_heartbeat = datetime.now()
        client.rounds_completed += 1
        client.consecutive_timeouts = 0
        client.status = ClientStatus.ACTIVE
        
        if accuracy is not None:
            client.model_accuracy = accuracy
        if quality_score is not None:
            client.data_quality_score = quality_score
        
        return True
    
    def detect_stale_clients(self) -> List[str]:
        """Detect and mark stale clients (no heartbeat)"""
        now = datetime.now()
        stale_clients = []
        
        for client_id, heartbeat in self.clients.items():
            if heartbeat.status == ClientStatus.DISCONNECTED:
                continue
            
            time_since_heartbeat = (now - heartbeat.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.idle_timeout:
                heartbeat.status = ClientStatus.DISCONNECTED
                heartbeat.consecutive_timeouts = 0
                print(f"[CLIENT_HEALTH] Marked disconnected: {heartbeat.client_name} (idle {time_since_heartbeat}s)")
                stale_clients.append(client_id)
            elif time_since_heartbeat > self.heartbeat_timeout:
                if heartbeat.status != ClientStatus.STALE:
                    heartbeat.status = ClientStatus.STALE
                    heartbeat.consecutive_timeouts += 1
                    print(f"[CLIENT_HEALTH] Marked stale: {heartbeat.client_name} (timeout #{heartbeat.consecutive_timeouts})")
                    stale_clients.append(client_id)
        
        return stale_clients
    
    def remove_dead_clients(self) -> List[str]:
        """Remove clients exceeding max consecutive timeouts"""
        dead_clients = []
        
        for client_id, heartbeat in list(self.clients.items()):
            if heartbeat.consecutive_timeouts >= self.max_consecutive_timeouts:
                print(f"[CLIENT_HEALTH] Removing dead client: {heartbeat.client_name} ({heartbeat.consecutive_timeouts} timeouts)")
                dead_clients.append(client_id)
                del self.clients[client_id]
        
        return dead_clients
    
    def get_active_clients(self, min_accuracy: float = None) -> Dict[str, ClientHeartbeat]:
        """Get active clients, optionally filtered by min accuracy"""
        active = {
            cid: hb for cid, hb in self.clients.items() 
            if hb.status == ClientStatus.ACTIVE
        }
        
        if min_accuracy is not None:
            active = {
                cid: hb for cid, hb in active.items()
                if hb.model_accuracy >= min_accuracy
            }
        
        return active
    
    def get_client_status(self, client_id: str) -> Optional[Dict]:
        """Get detailed status of specific client"""
        if client_id not in self.clients:
            return None
        
        hb = self.clients[client_id]
        now = datetime.now()
        
        return {
            'client_name': hb.client_name,
            'status': hb.status.value,
            'time_since_heartbeat': (now - hb.last_heartbeat).total_seconds(),
            'time_since_update': (now - hb.last_update).total_seconds(),
            'rounds_completed': hb.rounds_completed,
            'model_accuracy': hb.model_accuracy,
            'data_quality': hb.data_quality_score,
            'consecutive_timeouts': hb.consecutive_timeouts
        }
    
    def get_all_clients_status(self) -> List[Dict]:
        """Get status of all clients"""
        now = datetime.now()
        statuses = []
        
        for client_id, hb in self.clients.items():
            statuses.append({
                'client_id': client_id[:12] + '...',
                'client_name': hb.client_name,
                'status': hb.status.value,
                'time_since_update_sec': (now - hb.last_update).total_seconds(),
                'rounds_completed': hb.rounds_completed,
                'accuracy': round(hb.model_accuracy, 4),
                'data_quality': round(hb.data_quality_score, 4)
            })
        
        return statuses
    
    def get_health_summary(self) -> Dict:
        """Get health summary"""
        now = datetime.now()
        active = sum(1 for hb in self.clients.values() if hb.status == ClientStatus.ACTIVE)
        stale = sum(1 for hb in self.clients.values() if hb.status == ClientStatus.STALE)
        idle = sum(1 for hb in self.clients.values() if hb.status == ClientStatus.DISCONNECTED)
        
        total_rounds = sum(hb.rounds_completed for hb in self.clients.values())
        avg_accuracy = sum(hb.model_accuracy for hb in self.clients.values()) / len(self.clients) if self.clients else 0
        
        return {
            'total_clients': len(self.clients),
            'active_clients': active,
            'stale_clients': stale,
            'disconnected_clients': idle,
            'total_rounds_completed': total_rounds,
            'avg_client_accuracy': round(avg_accuracy, 4),
            'clients': self.get_all_clients_status()
        }

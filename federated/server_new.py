import socket
import json
import pickle
import numpy as np
from threading import Thread, Lock
from collections import defaultdict
import logging
import os

class FederatedServer:
    def __init__(self, host='localhost', port=8082):
        self.host = host
        self.port = port
        self.clients = []
        self.model_params = None
        self.current_round = 0
        self.metrics_lock = Lock()
        
        # Metrics storage with thread-safe access
        self.metrics = {
            'round': 0,
            'accuracy': [],
            'fairness_score': [],
            'fairness': [],
            'communication': [],
            'robustness': [],
            'active_clients': 0
        }
        
        # Client state tracking
        self.client_states = defaultdict(lambda: {'connected': False, 'last_update': 0})
        self.client_lock = Lock()
        # Configure logging to file and console
        log_path = os.path.join(os.path.dirname(__file__), 'server.log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
        logging.info(f"[SERVER] Initialized on {self.host}:{self.port}")

    def log(self, msg):
        try:
            logging.info(msg)
        except Exception:
            print(msg)
    
    def update_metrics(self, client_metrics):
        """Update global metrics with new client results"""
        with self.metrics_lock:
            round_num = client_metrics['round']
            if round_num > self.current_round:
                self.current_round = round_num
            
            # Update metrics arrays with new values
            self.metrics['accuracy'].append(client_metrics['accuracy'])
            self.metrics['fairness_score'].append(client_metrics['fairness_score'])
            self.metrics['fairness'].append(client_metrics['fairness'])
            self.metrics['communication'].append(0.95)  # Example communication quality score
            self.metrics['robustness'].append(0.90)     # Example robustness score
            
            # Update round number and client count
            self.metrics['round'] = self.current_round
            with self.client_lock:
                active_clients = sum(1 for state in self.client_states.values() if state['connected'])
                self.metrics['active_clients'] = active_clients
            
            # Print the latest metrics
            latest_idx = -1  # Get the most recent values
            self.log(f"[SERVER] Updated metrics for round {self.current_round}: Accuracy={self.metrics['accuracy'][latest_idx]:.4f} FairFinance={self.metrics['fairness_score'][latest_idx]:.4f} ActiveClients={self.metrics['active_clients']}")
    
    def start(self):
        try:
            print(f"[SERVER] Attempting to start server on {self.host}:{self.port}")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Try to bind to the port
            try:
                server_socket.bind((self.host, self.port))
            except socket.error as e:
                print(f"[SERVER] Failed to bind to port {self.port}. Error: {e}")
                print("[SERVER] This might mean another process is using this port.")
                return
            
            server_socket.listen(5)
            print(f"[SERVER] Successfully listening on {self.host}:{self.port}")
            
            while True:
                try:
                    print("[SERVER] Waiting for client connections...")
                    client_socket, addr = server_socket.accept()
                    print(f"[SERVER] Connected to {addr}")
                    client_thread = Thread(target=self.handle_client, args=(client_socket, addr))
                    client_thread.start()
                except Exception as e:
                    print(f"[SERVER] Error accepting connection: {e}")
        except Exception as e:
            print(f"[SERVER] Critical error starting server: {e}")
        finally:
            print("[SERVER] Shutting down server...")
            try:
                server_socket.close()
            except:
                pass
    
    def recv_msg(self, sock):
        try:
            # Get the size of the message first
            size_data = sock.recv(4)
            if not size_data:
                return None
            size = int.from_bytes(size_data, 'big')
            
            # Receive the full message in chunks
            data = bytearray()
            while len(data) < size:
                chunk = sock.recv(min(size - len(data), 8192))
                if not chunk:
                    return None
                data.extend(chunk)
            return data
        except Exception as e:
            print(f"[SERVER] Error receiving message: {e}")
            return None
    
    def send_msg(self, sock, data):
        try:
            # Send the size first
            size = len(data)
            sock.send(size.to_bytes(4, 'big'))
            # Send the data
            sock.sendall(data)
            return True
        except Exception as e:
            print(f"[SERVER] Error sending message: {e}")
            return False
    
    def handle_client(self, client_socket, addr):
        try:
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            # Wait for initial message to determine client type
            data = self.recv_msg(client_socket)
            if not data:
                return
                
            message = pickle.loads(data)
            self.log(f"[SERVER] Initial message from {addr}: {message}")

            if message.get('type') == 'get_metrics':
                self.handle_dashboard(client_socket)
            else:
                self.handle_training_client(client_socket, addr, message)
                
        except Exception as e:
            print(f"[SERVER] Error handling client {addr}: {e}")
        finally:
            with self.client_lock:
                self.client_states[addr]['connected'] = False
            client_socket.close()
            print(f"[SERVER] Client {addr} disconnected")
    
    def handle_dashboard(self, client_socket):
        self.log("[SERVER] Dashboard client connected")
        try:
            # Send current metrics immediately on connect so the dashboard
            # has data without waiting for the first poll.
            with self.metrics_lock:
                initial_resp = {
                    'type': 'metrics',
                    'round': self.metrics.get('round', 0),
                    'accuracy': list(self.metrics.get('accuracy', [])),
                    'fairness_score': list(self.metrics.get('fairness_score', [])),
                    'fairness': list(self.metrics.get('fairness', [])),
                    'communication': list(self.metrics.get('communication', [])),
                    'robustness': list(self.metrics.get('robustness', [])),
                    'active_clients': self.metrics.get('active_clients', 0)
                }
            self.log(f"[SERVER] Initial metrics sent to dashboard: {initial_resp}")
            try:
                self.send_msg(client_socket, pickle.dumps(initial_resp))
            except Exception as e:
                self.log(f"[SERVER] Failed to send initial metrics: {e}")

            while True:
                data = self.recv_msg(client_socket)
                if not data:
                    break

                try:
                    message = pickle.loads(data)
                except Exception as e:
                    self.log(f"[SERVER] Failed to decode dashboard message: {e}")
                    continue

                if message.get('type') == 'get_metrics':
                    with self.metrics_lock:
                        response = {
                            'type': 'metrics',
                            'round': self.metrics.get('round', 0),
                            'accuracy': list(self.metrics.get('accuracy', [])),
                            'fairness_score': list(self.metrics.get('fairness_score', [])),
                            'fairness': list(self.metrics.get('fairness', [])),
                            'communication': list(self.metrics.get('communication', [])),
                            'robustness': list(self.metrics.get('robustness', [])),
                            'active_clients': self.metrics.get('active_clients', 0)
                        }
                    self.log(f"[SERVER] Sending metrics to dashboard: {response}")
                    try:
                        self.send_msg(client_socket, pickle.dumps(response))
                        self.log(f"[SERVER] Sent metrics to dashboard: Round={self.metrics.get('round', 0)}")
                    except Exception as e:
                        self.log(f"[SERVER] Failed to send metrics to dashboard: {e}")
        except Exception as e:
            self.log(f"[SERVER] Error handling dashboard: {e}")
    
    def handle_training_client(self, client_socket, addr, initial_message):
        print(f"[SERVER] Training client {addr} connected")
        with self.client_lock:
            self.client_states[addr]['connected'] = True
        
        try:
            while True:
                if initial_message:
                    message = initial_message
                    initial_message = None
                else:
                    data = self.recv_msg(client_socket)
                    if not data:
                        break
                    message = pickle.loads(data)
                
                if message['type'] == 'model_update':
                    self.log(f"[SERVER] Received model_update from {addr}: keys={list(message.keys())}")
                    # Handle model parameter aggregation
                    incoming_params = message.get('params', {}) or {}
                    # If server has no params yet, initialize only if incoming has real params
                    if not self.model_params:
                        if incoming_params:
                            self.model_params = incoming_params
                            self.log(f"[SERVER] Initialized global model params with keys: {list(self.model_params.keys())}")
                        else:
                            self.log("[SERVER] Received empty params; keeping global model uninitialized")
                    else:
                        # Aggregate matching keys and add new keys if present
                        for key, val in incoming_params.items():
                            if key in self.model_params:
                                try:
                                    # For tensor-like values, try simple average
                                    self.model_params[key] = (self.model_params[key] + val) / 2
                                except Exception:
                                    self.model_params[key] = val
                            else:
                                self.model_params[key] = val
                    
                    # Update metrics with client results
                    client_metrics = {
                        'round': message.get('round', self.current_round + 1),
                        'accuracy': float(message.get('accuracy', 0.0)),
                        'fairness_score': float(message.get('fairness_score', 0.0)),
                        'fairness': float(message.get('fairness', 0.0))
                    }
                    self.update_metrics(client_metrics)
                    
                    # Send updated model back to client
                    response = {
                        'type': 'model_update',
                        'params': self.model_params if self.model_params else {},
                        'round': self.current_round
                    }
                    self.log(f"[SERVER] Sending model_update to client {addr}: params_keys={list(response['params'].keys())}")
                    self.send_msg(client_socket, pickle.dumps(response))
                    
        except Exception as e:
            print(f"[SERVER] Error handling training client {addr}: {e}")

if __name__ == "__main__":
    server = FederatedServer()
    server.start()
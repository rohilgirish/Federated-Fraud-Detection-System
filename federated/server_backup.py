import socket
import pickle
from threading import Thread

class SimpleFederatedServer:
    def __init__(self, host='localhost', port=8085):
        self.host = host
        self.port = port
        self.current_round = 1
        # Initialize with sample data
        self.metrics = {
            'accuracy': [0.75],
            'fairness_score': [0.85],
            'fairness': [0.80],
            'communication': [0.95],
            'robustness': [0.90]
        }
    
    def handle_client(self, client_socket):
        print("[SERVER] New client connected")
        try:
            while True:
                # Get message size
                size_data = client_socket.recv(4)
                if not size_data:
                    break
                size = int.from_bytes(size_data, 'big')
                
                # Get message data
                data = bytearray()
                while len(data) < size:
                    chunk = client_socket.recv(min(size - len(data), 8192))
                    if not chunk:
                        break
                    data.extend(chunk)
                
                if not data:
                    break
                    
                message = pickle.loads(data)
                print(f"[SERVER] Received message: {message}")
                
                if message['type'] == 'get_metrics':
                    response = {
                        'type': 'metrics',
                        'round': self.current_round,
                        'accuracy': self.metrics['accuracy'][-1],
                        'fairness_score': self.metrics['fairness_score'][-1],
                        'fairness': self.metrics['fairness'][-1],
                        'communication': self.metrics['communication'][-1],
                        'robustness': self.metrics['robustness'][-1],
                        'active_clients': 1
                    }
                    print(f"[SERVER] Sending response: {response}")
                    
                    # Send response size
                    response_data = pickle.dumps(response)
                    client_socket.send(len(response_data).to_bytes(4, 'big'))
                    # Send response data
                    client_socket.sendall(response_data)
                    
                    # Update metrics for next round (for testing)
                    self.current_round += 1
                    for metric in self.metrics:
                        self.metrics[metric].append(min(1.0, self.metrics[metric][-1] + 0.05))
                    
        except Exception as e:
            print(f"[SERVER] Error handling client: {e}")
        finally:
            print("[SERVER] Client disconnected")
            client_socket.close()
    
    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"[SERVER] Listening on {self.host}:{self.port}")
        
        try:
            while True:
                client_socket, addr = server_socket.accept()
                Thread(target=self.handle_client, args=(client_socket,)).start()
        except KeyboardInterrupt:
            print("[SERVER] Shutting down")
        finally:
            server_socket.close()

if __name__ == "__main__":
    server = SimpleFederatedServer()
    server.start()
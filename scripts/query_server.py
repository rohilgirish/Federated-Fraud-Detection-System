import socket
import pickle

HOST = 'localhost'
PORT = 8085

def send_msg(sock, data_bytes):
    sock.send(len(data_bytes).to_bytes(4, 'big'))
    sock.sendall(data_bytes)

def recv_msg(sock):
    size_data = sock.recv(4)
    if not size_data:
        return None
    size = int.from_bytes(size_data, 'big')
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), 8192))
        if not chunk:
            return None
        data.extend(chunk)
    return data

with socket.create_connection((HOST, PORT), timeout=5) as s:
    req = {'type': 'get_metrics'}
    send_msg(s, pickle.dumps(req))
    resp_bytes = recv_msg(s)
    if not resp_bytes:
        print('No response')
    else:
        resp = pickle.loads(resp_bytes)
        clients = resp.get('clients', [])
        print(f"Server reports {len(clients)} clients:")
        for c in clients:
            print(f" - {c.get('client_id')} | {c.get('client_name')} | status={c.get('status')} | last_update={c.get('last_update')} | rounds={c.get('rounds_completed')} | avg_acc={c.get('avg_accuracy')}")

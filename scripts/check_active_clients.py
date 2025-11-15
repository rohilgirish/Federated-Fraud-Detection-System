import socket, pickle
from datetime import datetime, timedelta
HOST='localhost'; PORT=8085
try:
    s=socket.create_connection((HOST,PORT), timeout=5)
    req={'type':'get_metrics'}
    msg=pickle.dumps(req)
    s.send(len(msg).to_bytes(4,'big'))
    s.sendall(msg)
    size_data=s.recv(4)
    if size_data:
        size=int.from_bytes(size_data,'big')
        data=b''
        while len(data)<size:
            data+=s.recv(min(8192,size-len(data)))
        resp=pickle.loads(data)
        clients=resp.get('clients',[])
        now=datetime.now()
        timeout=timedelta(seconds=60)
        active=[c for c in clients if c.get('last_update') and (now - datetime.strptime(c.get('last_update'), '%Y-%m-%d %H:%M:%S'))<=timeout]
        print(f"Total clients from server: {len(clients)}")
        print(f"Active clients (last 60s): {len(active)}")
        for i,c in enumerate(active,1):
            name=c.get('client_name') or f'Client-{i}'
            print(i, c.get('client_id'), '=>', name, 'rounds=', c.get('rounds_completed'), 'avg_acc=', c.get('avg_accuracy'))
    else:
        print('No response')
    s.close()
except Exception as e:
    print('Error querying server:', e)

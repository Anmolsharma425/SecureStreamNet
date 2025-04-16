import socket
import threading
import sys
import base64

clients = {}

def broadcast_client_list() -> None:
    """Send the updated list of clients and their public keys to all clients."""
    client_list = "\n".join([f"{name}|{base64.b64encode(info['public_key']).decode()}" for name, info in clients.items()])
    message = f"CLIENT_LIST:{client_list}"
    message_bytes = message.encode()
    length = len(message_bytes)
    for client in clients.values():
        try:
            client['socket'].sendall(length.to_bytes(4, "big") + message_bytes)
        except Exception as e:
            print(f"Error sending client list to {client}: {e}")

def broadcast_message(message: str) -> None:
    """Broadcast encrypted message to all connected clients."""
    full_message = f"MESSAGE:{message}"
    message_bytes = full_message.encode()
    length = len(message_bytes)
    for client in clients.values():
        try:
            client['socket'].sendall(length.to_bytes(4, "big") + message_bytes)
        except Exception as e:
            print(f"Error broadcasting to {client}: {e}")

def handle_client(client_socket: socket.socket, client_address: tuple) -> None:
    """Handle communication with an individual client."""
    try:
        name_length = int.from_bytes(client_socket.recv(4), "big")
        client_name = client_socket.recv(name_length).decode()
        key_length = int.from_bytes(client_socket.recv(4), "big")
        public_key = client_socket.recv(key_length)
        
        # Check for unique name and client limit
        if client_name in clients:
            client_socket.sendall("Name already taken".encode())
            client_socket.close()
            return
        if len(clients) >= 5:
            client_socket.sendall("Server is full".encode())
            client_socket.close()
            return
        
        # Add client and send confirmation
        clients[client_name] = {'socket': client_socket, 'public_key': public_key}
        client_socket.sendall("OK".encode())
        print(f"[NEW CONNECTION] {client_name} connected from {client_address}")
        
        broadcast_client_list()
        
        while True:
            recipient_name = client_socket.recv(1024).decode()
            encrypted_message = client_socket.recv(4096).decode()
            if not recipient_name or not encrypted_message:
                break
            print(f"[MESSAGE RECEIVED] {client_name} to {recipient_name}: {encrypted_message[:50]}...")
            broadcast_message(f"{client_name}:{recipient_name}:{encrypted_message}")
    except ConnectionResetError:
        print(f"[DISCONNECTED] {client_name} disconnected.")
    finally:
        if client_name in clients:
            del clients[client_name]
            broadcast_client_list()
        client_socket.close()

def start_server(host: str, port: int) -> None:
    """Initialize and start the server to accept client connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[SERVER STARTED] Listening on {host}:{port}")
    
    try:
        while True:
            client_socket, client_address = server_socket.accept()
            client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
            client_thread.start()
    except KeyboardInterrupt:
        print("[SERVER SHUTDOWN] Closing server...")
    finally:
        server_socket.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python server.py <HOST> <PORT>")
        sys.exit(1)
    
    HOST = sys.argv[1]
    PORT = int(sys.argv[2])
    start_server(HOST, PORT)
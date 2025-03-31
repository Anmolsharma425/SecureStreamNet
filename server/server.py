import socket
import threading
import sys

# Dictionary to store client public keys and their socket objects
clients = {}

def handle_client(client_socket: socket.socket, client_address: tuple) -> None:
    """Handle communication with an individual client."""
    try:
        client_name = client_socket.recv(1024).decode()
        public_key = client_socket.recv(4096).decode()
        
        clients[client_name] = {'socket': client_socket, 'public_key': public_key}
        print(f"[NEW CONNECTION] {client_name} connected from {client_address}")
        
        # Notify all clients about the updated client list
        broadcast_client_list()
        
        while True:
            encrypted_message = client_socket.recv(4096).decode()
            if not encrypted_message:
                break
            print(f"[MESSAGE RECEIVED] {client_name}: {encrypted_message[:50]}...")
            
            # Forward the encrypted message to all clients
            broadcast_message(client_name, encrypted_message)
    except ConnectionResetError:
        print(f"[DISCONNECTED] {client_name} disconnected.")
    finally:
        client_socket.close()
        remove_client(client_name)

def broadcast_client_list() -> None:
    """Send the updated list of clients and their public keys to all clients."""
    client_list = "\n".join([f"{name}:{info['public_key']}" for name, info in clients.items()])
    for client in clients.values():
        client['socket'].sendall(client_list.encode())

def broadcast_message(sender: str, encrypted_message: str) -> None:
    """Broadcast encrypted messages to all clients."""
    for client_name, client_info in clients.items():
        if client_name != sender:  # Don't send the message back to the sender
            client_info['socket'].sendall(f"{sender}:{encrypted_message}".encode())

def remove_client(client_name: str) -> None:
    """Remove a client from the dictionary when they disconnect."""
    if client_name in clients:
        del clients[client_name]
        broadcast_client_list()

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
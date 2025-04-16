import socket
import threading
import sys
import base64
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

clients = {}

def connect_to_server(server_ip: str, server_port: int, client_name: str) -> tuple:
    """Connect the client to the server and send the client’s name and public key."""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    serialized_public_key = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    client_socket.sendall(len(client_name).to_bytes(4, "big") + client_name.encode())
    client_socket.sendall(len(serialized_public_key).to_bytes(4, "big") + serialized_public_key)
    
    response = client_socket.recv(1024).decode()
    if response != "OK":
        print(f"Connection failed: {response}")
        client_socket.close()
        sys.exit(1)
    
    print(f"{client_name} connected to server and sent public key.")
    return client_socket, private_key

def update_client_list(client_list_data: str) -> None:
    """Update the local client list from server data."""
    global clients
    client_list = client_list_data.split("\n")
    clients = {}
    for item in client_list:
        if item:
            name, key = item.split("|", 1)
            clients[name] = base64.b64decode(key.encode())
    print(f"Updated client list: {list(clients.keys())}")

def handle_message(message_data: str, private_key, client_name: str) -> None:
    """Process incoming messages and decrypt if intended for this client."""
    parts = message_data.split(":", 2)
    if len(parts) == 3:
        sender, recipient, encrypted_message_hex = parts
        if recipient == client_name:
            encrypted_message = bytes.fromhex(encrypted_message_hex)
            try:
                decrypted_message = private_key.decrypt(
                    encrypted_message,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                print(f"Message from {sender}: {decrypted_message.decode()}")
            except Exception as e:
                print(f"Error decrypting message: {e}")
        else:
            print(f"Ignoring message for {recipient}")
    else:
        print("Invalid message format")

def receive_messages(client_socket: socket.socket, private_key, client_name: str) -> None:
    """Receive and process all server messages."""
    while True:
        try:
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, "big")
            message = client_socket.recv(length).decode()
            if message.startswith("CLIENT_LIST:"):
                client_list_data = message[len("CLIENT_LIST:"):]
                update_client_list(client_list_data)
            elif message.startswith("MESSAGE:"):
                message_data = message[len("MESSAGE:"):]
                handle_message(message_data, private_key, client_name)
            else:
                print("Unknown message type")
        except Exception as e:
            print(f"Error receiving message: {e}")
            break

def send_encrypted_message(client_socket: socket.socket, recipient: str, message: str) -> None:
    """Encrypt and send a message to the server."""
    if recipient not in clients:
        print(f"Recipient {recipient} not found in client list.")
        return
    
    recipient_public_key = serialization.load_pem_public_key(clients[recipient])
    encrypted_message = recipient_public_key.encrypt(
        message.encode(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    client_socket.sendall(recipient.encode())
    client_socket.sendall(encrypted_message.hex().encode())
    print(f"Encrypted message sent to {recipient}.")

def start_client(server_ip: str, server_port: int, client_name: str) -> None:
    """Start the client with a simple interface."""
    client_socket, private_key = connect_to_server(server_ip, server_port, client_name)
    
    threading.Thread(target=receive_messages, args=(client_socket, private_key, client_name), daemon=True).start()
    
    while True:
        recipient = input("Enter recipient (e.g., client1 or client2): ")
        message = input("Enter message: ")
        send_encrypted_message(client_socket, recipient, message)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python client.py <SERVER_IP> <SERVER_PORT> <CLIENT_NAME>")
        sys.exit(1)
    
    SERVER_IP = sys.argv[1]
    SERVER_PORT = int(sys.argv[2])
    CLIENT_NAME = sys.argv[3]
    start_client(SERVER_IP, SERVER_PORT, CLIENT_NAME)
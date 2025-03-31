import socket
import pickle
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# Dictionary to store received clients and their public keys
clients = {}

def connect_to_server(server_ip: str, server_port: int) -> socket.socket:
    """Connect the client to the server and send the client’s public key."""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    
    # Generate RSA key pair
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    
    # Serialize the public key
    serialized_public_key = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    # Send public key to the server
    client_socket.sendall(serialized_public_key)
    
    print("Connected to server and sent public key.")
    return client_socket, private_key

def receive_broadcasted_clients(client_socket: socket.socket) -> None:
    """Update the local dictionary with the latest client list from the server."""
    try:
        data = client_socket.recv(4096)
        if data:
            global clients
            clients = pickle.loads(data)
            print("Updated client list:", clients)
    except Exception as e:
        print(f"Error receiving client list: {e}")

def send_encrypted_message(client_socket: socket.socket, recipient: str, message: str) -> None:
    """Encrypt a message using the recipient's public key and send it to the server."""
    if recipient not in clients:
        print("Recipient not found in client list.")
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
    
    client_socket.sendall(pickle.dumps((recipient, encrypted_message)))
    print("Encrypted message sent.")

def receive_and_decrypt_message(client_socket: socket.socket, private_key) -> None:
    """Receive encrypted messages and decrypt if the client is the intended recipient."""
    try:
        data = client_socket.recv(4096)
        if data:
            sender, encrypted_message = pickle.loads(data)
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
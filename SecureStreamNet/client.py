import socket
import threading
import sys
import base64
import cv2
import pickle
import struct
import time
import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

clients = {}

def connect_to_server(client_socket: socket.socket, client_name: str) -> tuple:    
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
    return private_key

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

# Receive video
def receive_video(client_socket):
    """Receive and display video frames from the server with precise timing and full-screen mode."""
    target_resolution = (1280, 720)  # 720p display
    
    cv2.namedWindow("Video Stream", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Receive FPS from server
    fps_data = b""
    while len(fps_data) < 4:
        packet = client_socket.recv(4 - len(fps_data))
        if not packet:
            print("Server closed connection before sending FPS")
            return
        fps_data += packet

    target_fps = struct.unpack("f", fps_data)[0]
    if target_fps <= 0 or target_fps > 100:
        print(f"[WARNING] Invalid FPS received ({target_fps}), defaulting to 30.")
        target_fps = 30.0
    frame_time = 1.0 / target_fps


    try:
        frame_count = 0
        start_time = time.time()
        next_frame_time = start_time

        while True:
            size_data = b""
            while len(size_data) < 8:
                packet = client_socket.recv(8 - len(size_data))
                if not packet:
                    print("Server closed connection")
                    return
                size_data += packet

            size = struct.unpack("Q", size_data)[0]

            frame_data = b""
            while len(frame_data) < size:
                packet = client_socket.recv(min(4096, size - len(frame_data)))
                if not packet:
                    print("Incomplete frame received")
                    return
                frame_data += packet

            try:
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is None:
                    print("Failed to decode frame")
                    continue

                resized_frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
                cv2.imshow("Video Stream", resized_frame)

                frame_count += 1
                next_frame_time += frame_time
                current_time = time.time()
                sleep_time = next_frame_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error decoding frame: {e}")
                continue
    finally:
        cv2.destroyAllWindows()
        client_socket.close()


# receive service
def receive_services(sock):
    services = sock.recv(1024).decode().split(',')
    print("Available services:", services)
    return services

# choose service
def choose_service(sock):
    service = input("Choose a service: ").strip()
    sock.sendall(service.encode())
    return service

# code to choose video
def choose_video(sock):
    video_list = sock.recv(2048).decode().split(',')
    print("Available videos:")
    for idx, vid in enumerate(video_list):
        print(f"{idx+1}. {vid}")
    choice = int(input("Enter the video number to play: ")) - 1
    chosen_video = video_list[choice].strip()
    sock.sendall(chosen_video.encode())
    return chosen_video

def start_client(server_ip: str, server_port: int, client_name: str) -> None:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    # Receive the message from the server
    response_length = int.from_bytes(client_socket.recv(4), "big")
    response = client_socket.recv(response_length).decode()

    # Send the message to the server
    service_type = input(response)
    service_bytes = service_type.encode()
    length_bytes = len(service_bytes).to_bytes(4, 'big')
    client_socket.sendall(length_bytes + service_bytes)
    while True:
        if service_type == "message":
            private_key = connect_to_server(client_socket, client_name)
            threading.Thread(target=receive_messages, args=(client_socket, private_key, client_name), daemon=True).start()
    
            while True:
                recipient = input("Enter recipient : ")
                message = input("Enter message: ")
                send_encrypted_message(client_socket, recipient, message)
    
        elif service_type == "video":
            print(f"Connected to server at {server_ip}:{server_port}")
            client_socket.sendall(client_name.encode())
            services = receive_services(client_socket)
            selected_service = choose_service(client_socket)
            if selected_service == "get_video":
                chosen_video = choose_video(client_socket)
                print(f"Streaming video: {chosen_video}")
                receive_video(client_socket)
                
        else:
            print("Unknow services")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python client.py <SERVER_IP> <SERVER_PORT> <CLIENT_NAME>")
        sys.exit(1)
    
    SERVER_IP = sys.argv[1]
    SERVER_PORT = int(sys.argv[2])
    CLIENT_NAME = sys.argv[3]
    start_client(SERVER_IP, SERVER_PORT, CLIENT_NAME)
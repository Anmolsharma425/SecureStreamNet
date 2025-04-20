import socket
import threading
import sys
import base64
import os
import cv2
import pickle
import struct
import time
import numpy as np

clients = {}

# BroadCasting the info of the new client to all the remaining clients
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

# Broadcasting the encrypted message to all the clients
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

# directory for the video
video_folder = "./videos"

def validate_video_file(video_path):
    """Validate that a video file is readable and has frames, return FPS and frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False, 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    # Use fallback FPS if invalid
    if raw_fps <= 0:
        fps = 30.0
        print(f"[WARNING] Invalid FPS for {video_path}, using fallback: {fps} FPS")
    else:
        fps = raw_fps
    cap.release()
    return frame_count > 0, frame_count, fps

# logics for the video streaming
def stream_video(client_socket, base_name):
    """Stream video frames to the client in a round-robin fashion with synchronized FPS."""
    # Construct paths for the three quality versions
    variants = [os.path.join(video_folder, f"{base_name}_{q}.mp4") for q in ['240p','360p', '720p', '1080p']]
    
    # Validate all video files and collect FPS
    valid_variants = []
    fps_values = []
    for v in variants:
        is_valid, frame_count, fps = validate_video_file(v)
        if is_valid:
            valid_variants.append(v)
            fps_values.append(fps)
    
    if not valid_variants:
        print(f"[ERROR] No valid video files for '{base_name}'")
        client_socket.sendall(b"ERROR: No valid video files available")
        return
    if len(valid_variants) < 3:
        print(f"[WARNING] Only {len(valid_variants)} of 3 quality versions available for '{base_name}'")
    
    # Select the minimum FPS for synchronization (downsampling)
    selected_fps = min(fps_values) if fps_values else 30.0
    frame_time = 1.0 / selected_fps  # Time per frame in seconds
    print(f"[INFO] Synchronized FPS: {selected_fps} for '{base_name}'")
    
    # Send FPS to the client (4 bytes, float)
    fps_data = struct.pack("f", selected_fps)
    try:
        client_socket.sendall(fps_data)
        print(f"[INFO] Sent FPS: {selected_fps} to client")
    except Exception as e:
        print(f"[ERROR] Failed to send FPS to client: {e}")
        return
    
    # Open the video files and calculate frame skip ratios
    caps = [cv2.VideoCapture(v) for v in valid_variants]
    frame_skip_ratios = [fps / selected_fps for fps in fps_values]  # Ratio to skip frames
    frame_counters = [0] * len(caps)  # Track frame positions for skipping
    
    try:
        frame_count = 0
        start_time = time.time()  # For measuring actual FPS
        active_caps = caps.copy()  # Track active video captures
        while active_caps:
            frame_start_time = time.time()  # Track individual frame send time
            # Read one frame from each active video in sequence
            for i, cap in enumerate(active_caps[:]):  # Iterate over a copy
                try:
                    # Calculate which frame to read based on FPS ratio
                    frame_counters[i] += frame_skip_ratios[i]
                    target_frame = int(frame_counters[i])
                    
                    # Skip frames to match target FPS (downsampling)
                    while cap.get(cv2.CAP_PROP_POS_FRAMES) < target_frame:
                        ret, _ = cap.read()
                        if not ret:
                            active_caps.remove(cap)
                            cap.release()
                            break
                    if cap not in active_caps:
                        continue
                    
                    # Read the target frame
                    ret, frame = cap.read()
                    if not ret:
                        active_caps.remove(cap)
                        cap.release()
                        continue
                    
                    # Compress frame as JPEG with lower quality for speed
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Reduced from 90 to 80
                    result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                    if not result:
                        print(f"[ERROR] Failed to encode frame from {valid_variants[i]}")
                        continue
                    data = encoded_frame.tobytes()
                    
                    # Send the compressed frame
                    message_size = struct.pack("Q", len(data))
                    try:
                        client_socket.sendall(message_size + data)
                        frame_count += 1
                    except (BrokenPipeError, ConnectionResetError):
                        print(f"[ERROR] Client disconnected during frame send")
                        return
                    
                    # Log actual FPS every 30 frames
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        actual_fps = frame_count / elapsed
                    
                    # Control frame rate
                    elapsed = time.time() - frame_start_time
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                except Exception as e:
                    print(f"[ERROR] Failed to read frame from {valid_variants[i]}: {e}")
                    active_caps.remove(cap)
                    cap.release()
                    continue
            
            # Break if no videos remain active
            if not active_caps:
                break
    except Exception as e:
        print(f"[ERROR] Streaming error for '{base_name}': {e}")
    finally:
        # Release any remaining video capture objects
        for cap in caps:
            if cap.isOpened():
                cap.release()
        print(f"[INFO] Finished streaming '{base_name}'")

# Handle client message request
def handle_client_message(client_socket: socket.socket, client_address: tuple) -> None:
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

def get_available_video_names():
    files = os.listdir(video_folder)
    base_names = set()
    for f in files:
        if f.endswith('.mp4') and '_' in f:
            base = f.split('_')[0]
            base_names.add(base)
    return list(base_names)

def handle_client_video(client_socket, address):
    print(f"Client connected: {address}")
    try:
        client_name = client_socket.recv(1024).decode().strip()
        print(f"[INFO] Client name: {client_name}")

        # Send available services
        services = ["get_video"]
        print("HII")
        client_socket.sendall(','.join(services).encode())

        # Receive selected service
        selected_service = client_socket.recv(1024).decode().strip()
        print(f"Client selected service: {selected_service}")

        if selected_service == "get_video":
            video_names = get_available_video_names()
            client_socket.sendall(','.join(video_names).encode())

            # Receive chosen video
            chosen_video = client_socket.recv(1024).decode().strip()
            print(f"Client chose video: {chosen_video}")
            stream_video(client_socket, chosen_video)
        else:
            print("Unknown service requested.")
            client_socket.sendall(b"ERROR: Unsupported service.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        client_socket.close()
        print(f"Client disconnected: {address}")

# connecting client to the server based on the service
def start_server(host: str, port: int) -> None:
    """Initialize and start the server to accept client connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[SERVER STARTED] Listening on {host}:{port}")
    
    try:
        while True:
            try:
                client_socket, client_address = server_socket.accept()
                message = "Enter Service(message|video)"
                message_bytes = message.encode()
                client_socket.sendall(len(message_bytes).to_bytes(4, "big") + message_bytes)                
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        service_length = int.from_bytes(client_socket.recv(4), "big")
                        service_name = client_socket.recv(service_length).decode()
                        print(service_name)
                        
                        if service_name == "message":
                            client_thread = threading.Thread(target=handle_client_message, args=(client_socket, client_address))
                            client_thread.start()
                            break
                        elif service_name == "video":
                            client_thread = threading.Thread(target=handle_client_video, args=(client_socket, client_address))
                            client_thread.start()
                            break
                        else:
                            retry_count += 1
                            remaining = max_retries - retry_count
                            if remaining > 0:
                                client_socket.sendall(f"Invalid service. {remaining} attempts remaining. Please enter 'message' or 'video'.".encode())
                            else:
                                client_socket.sendall("Too many invalid attempts. Closing connection.".encode())
                                client_socket.close()
                    except ValueError:
                        retry_count += 1
                        remaining = max_retries - retry_count
                        if remaining > 0:
                            client_socket.sendall(f"Invalid service format. {remaining} attempts remaining. Please enter 'message' or 'video'.".encode())
                        else:
                            client_socket.sendall("Too many invalid attempts. Closing connection.".encode())
                            client_socket.close()
            except Exception as e:
                print(f"[ERROR] Client connection error: {e}")
                if 'client_socket' in locals():
                    client_socket.close()
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
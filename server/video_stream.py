import os
import cv2
import socket
import struct
import pickle

def list_available_videos() -> list:
    """Return a list of available videos and their resolutions."""
    video_folder = "videos"  # Directory containing videos
    if not os.path.exists(video_folder):
        return []
    
    video_files = []
    for file in os.listdir(video_folder):
        if file.endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(video_folder, file)
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_files.append((file, f"{width}x{height}"))
                cap.release()
    return video_files

def stream_video(client_socket: socket.socket, video_name: str) -> None:
    """Stream a video by mixing frames from different resolutions in sequence."""
    video_path = os.path.join("videos", video_name)
    if not os.path.exists(video_path):
        print(f"Video {video_name} not found.")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_name}.")
        return
    
    resolutions = [(640, 360), (1280, 720), (1920, 1080)]
    res_index = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize the frame sequentially based on predefined resolutions
            target_res = resolutions[res_index % len(resolutions)]
            res_index += 1
            frame = cv2.resize(frame, target_res)
            
            # Serialize the frame using pickle
            data = pickle.dumps(frame)
            message_size = struct.pack("Q", len(data))
            
            # Send data over the socket
            client_socket.sendall(message_size + data)
    except Exception as e:
        print(f"Error streaming video: {e}")
    finally:
        cap.release()
        client_socket.close()
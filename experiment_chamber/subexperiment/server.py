import socket
import threading
import sys
import tkinter as tk
from tkinter import messagebox, ttk
import base64
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

clients = {}
client_socket = None
private_key = None
client_name = None

def connect_to_server(server_ip: str, server_port: int, name: str) -> tuple:
    """Connect the client to the server and send the client’s name and public key."""
    global client_socket, private_key, client_name
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((server_ip, server_port))
        
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        serialized_public_key = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        client_socket.sendall(len(name).to_bytes(4, "big") + name.encode())
        client_socket.sendall(len(serialized_public_key).to_bytes(4, "big") + serialized_public_key)
        
        response = client_socket.recv(1024).decode()
        if response != "OK":
            raise Exception(f"Connection failed: {response}")
        
        client_name = name
        return True
    except Exception as e:
        messagebox.showerror("Connection Error", str(e))
        return False

def update_client_list(client_list_data: str) -> None:
    """Update the local client list from server data."""
    global clients
    client_list = client_list_data.split("\n")
    clients = {name: base64.b64decode(key.encode()) for name, key in (item.split("|", 1) for item in client_list if item)}
    client_listbox.delete(0, tk.END)
    for name in clients.keys():
        client_listbox.insert(tk.END, name)

def handle_message(message_data: str) -> None:
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
                chat_text.config(state=tk.NORMAL)
                chat_text.insert(tk.END, f"{sender}: {decrypted_message.decode()}\n")
                chat_text.config(state=tk.DISABLED)
            except Exception as e:
                chat_text.config(state=tk.NORMAL)
                chat_text.insert(tk.END, f"Error decrypting message from {sender}: {e}\n")
                chat_text.config(state=tk.DISABLED)

def receive_messages() -> None:
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
                root.after(0, update_client_list, client_list_data)
            elif message.startswith("MESSAGE:"):
                message_data = message[len("MESSAGE:"):]
                root.after(0, handle_message, message_data)
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Receive Error", f"Disconnected: {e}"))
            break

def send_encrypted_message() -> None:
    """Encrypt and send a message to the server."""
    recipient = client_listbox.get(tk.ACTIVE)
    message = message_entry.get()
    if not recipient:
        messagebox.showwarning("Input Error", "Please select a recipient.")
        return
    if not message:
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    if recipient not in clients:
        messagebox.showwarning("Recipient Error", f"{recipient} not found.")
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
    chat_text.config(state=tk.NORMAL)
    chat_text.insert(tk.END, f"You to {recipient}: {message}\n")
    chat_text.config(state=tk.DISABLED)
    message_entry.delete(0, tk.END)

def start_client(server_ip: str, server_port: int, name: str) -> None:
    """Start the client and connect to the server."""
    if connect_to_server(server_ip, server_port, name):
        connect_button.config(state=tk.DISABLED)
        send_button.config(state=tk.NORMAL)
        threading.Thread(target=receive_messages, daemon=True).start()

def on_connect():
    """Handle connect button click."""
    server_ip = ip_entry.get()
    server_port = port_entry.get()
    name = name_entry.get()
    if not server_ip or not server_port or not name:
        messagebox.showwarning("Input Error", "Please fill in all fields.")
        return
    try:
        port = int(server_port)
        threading.Thread(target=start_client, args=(server_ip, port, name), daemon=True).start()
    except ValueError:
        messagebox.showerror("Input Error", "Port must be a number.")

def on_closing():
    """Handle window close event."""
    if client_socket:
        client_socket.close()
    root.destroy()

# Set up the GUI
root = tk.Tk()
root.title("Secure Chat Client")
root.geometry("600x400")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Connection frame
connect_frame = ttk.Frame(root, padding="10")
connect_frame.pack(fill=tk.X)

ttk.Label(connect_frame, text="Server IP:").grid(row=0, column=0, padx=5, pady=5)
ip_entry = ttk.Entry(connect_frame)
ip_entry.grid(row=0, column=1, padx=5, pady=5)
ip_entry.insert(0, "127.0.0.1")

ttk.Label(connect_frame, text="Port:").grid(row=0, column=2, padx=5, pady=5)
port_entry = ttk.Entry(connect_frame)
port_entry.grid(row=0, column=3, padx=5, pady=5)
port_entry.insert(0, "5555")

ttk.Label(connect_frame, text="Name:").grid(row=0, column=4, padx=5, pady=5)
name_entry = ttk.Entry(connect_frame)
name_entry.grid(row=0, column=5, padx=5, pady=5)

connect_button = ttk.Button(connect_frame, text="Connect", command=on_connect)
connect_button.grid(row=0, column=6, padx=5, pady=5)

# Main chat frame
chat_frame = ttk.Frame(root, padding="10")
chat_frame.pack(fill=tk.BOTH, expand=True)

# Client list
ttk.Label(chat_frame, text="Connected Clients:").pack(anchor=tk.W)
client_listbox = tk.Listbox(chat_frame, height=5)
client_listbox.pack(fill=tk.X, pady=5)

# Chat display
ttk.Label(chat_frame, text="Chat:").pack(anchor=tk.W)
chat_text = tk.Text(chat_frame, height=10, state=tk.DISABLED)
chat_text.pack(fill=tk.BOTH, expand=True)

# Message input
message_frame = ttk.Frame(root, padding="10")
message_frame.pack(fill=tk.X)

ttk.Label(message_frame, text="Message:").grid(row=0, column=0, padx=5, pady=5)
message_entry = ttk.Entry(message_frame)
message_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
message_frame.columnconfigure(1, weight=1)

send_button = ttk.Button(message_frame, text="Send", command=send_encrypted_message, state=tk.DISABLED)
send_button.grid(row=0, column=2, padx=5, pady=5)

# Start the GUI
root.mainloop()
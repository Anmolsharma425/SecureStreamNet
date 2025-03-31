#!/bin/bash

# Set the host and port
HOST="127.0.0.1"
PORT=12345

# source venv/bin/activate

# Run the server script
python3 server/server.py "$HOST" "$PORT"

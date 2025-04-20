PYTHON = python
HOST = 127.0.0.1
PORT = 4000 
SERVER_FILE = server.py
CLIENT_FILE = client.py

# Default target (just shows usage)
all:
	@echo "Usage: make [server | client1 | client2 | clean]"
	@echo "  - make server: Start the server on $(HOST):$(PORT)"
	@echo "  - make client1: Start client1 connected to $(HOST):$(PORT)"
	@echo "  - make client2: Start client2 connected to $(HOST):$(PORT)"
	@echo "  - make clean: Remove Python cache files"

# Target to run the server
server:
	$(PYTHON) $(SERVER_FILE) $(HOST) $(PORT)

# Target to run client1
client1:
	$(PYTHON) $(CLIENT_FILE) $(HOST) $(PORT) client1

# Target to run client2
client2:
	$(PYTHON) $(CLIENT_FILE) $(HOST) $(PORT) client2

# Target to run client3
client3:
	$(PYTHON) $(CLIENT_FILE) $(HOST) $(PORT) client3

# Target to clean up Python cache files
clean:
	rm -rf __pycache__ *.pyc

.PHONY: all server client1 client2 client3 clean

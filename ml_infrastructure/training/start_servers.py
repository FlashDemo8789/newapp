"""
Start both ModelServer and SimpleAPI servers

This script starts both servers in separate threads and provides
a clean shutdown mechanism.
"""

import os
import time
import logging
import threading
import argparse
import signal
import socket
import sys
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port: int, max_attempts: int = 5) -> int:
    """Find an available port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        if not is_port_in_use(port):
            return port
    
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def start_model_server(host: str, port: int) -> threading.Thread:
    """Start the ModelServer in a separate thread"""
    from ml_infrastructure.training.serve_startup_model import start_model_server
    
    # Find an available port
    actual_port = find_available_port(port)
    logger.info(f"Starting ModelServer on {host}:{actual_port}")
    
    # Start the server in a thread
    thread = threading.Thread(
        target=start_model_server,
        args=(host, actual_port),
        daemon=True
    )
    thread.start()
    
    # Wait for the server to start
    for _ in range(5):
        if is_port_in_use(actual_port):
            logger.info(f"ModelServer is running on {host}:{actual_port}")
            break
        time.sleep(1)
    else:
        logger.warning(f"Could not confirm if ModelServer is running on {host}:{actual_port}")
    
    return thread

def start_simple_api(host: str, port: int) -> threading.Thread:
    """Start the SimpleAPI in a separate thread"""
    from ml_infrastructure.training.simple_api import start_server
    
    # Find an available port
    actual_port = find_available_port(port)
    logger.info(f"Starting SimpleAPI on {host}:{actual_port}")
    
    # Start the server in a thread
    thread = threading.Thread(
        target=start_server,
        args=(host, actual_port),
        daemon=True
    )
    thread.start()
    
    # Wait for the server to start
    for _ in range(5):
        if is_port_in_use(actual_port):
            logger.info(f"SimpleAPI is running on {host}:{actual_port}")
            break
        time.sleep(1)
    else:
        logger.warning(f"Could not confirm if SimpleAPI is running on {host}:{actual_port}")
    
    return thread

def test_servers(model_server_port: int, simple_api_port: int) -> Tuple[bool, bool]:
    """Test both servers to make sure they're working"""
    import requests
    
    # Test ModelServer
    model_server_ok = False
    try:
        response = requests.post(
            f"http://localhost:{model_server_port}/models/startup_success_predictor/predict",
            json={"stage": "seed", "sector": "fintech", "monthly_revenue": 50000},
            timeout=2
        )
        model_server_ok = response.status_code == 200
        logger.info(f"ModelServer test: {'SUCCESS' if model_server_ok else 'FAILED'}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error testing ModelServer: {e}")
    
    # Test SimpleAPI
    simple_api_ok = False
    try:
        # Health check
        health_response = requests.get(f"http://localhost:{simple_api_port}/health", timeout=2)
        
        # Prediction
        pred_response = requests.post(
            f"http://localhost:{simple_api_port}/predict",
            json={"stage": "seed", "sector": "fintech", "monthly_revenue": 50000},
            timeout=2
        )
        
        simple_api_ok = health_response.status_code == 200 and pred_response.status_code == 200
        logger.info(f"SimpleAPI test: {'SUCCESS' if simple_api_ok else 'FAILED'}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error testing SimpleAPI: {e}")
    
    return model_server_ok, simple_api_ok

def main():
    parser = argparse.ArgumentParser(description="Start both ML infrastructure servers")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the servers")
    parser.add_argument("--model-server-port", type=int, default=5000, help="Port for ModelServer")
    parser.add_argument("--simple-api-port", type=int, default=5001, help="Port for SimpleAPI")
    parser.add_argument("--test", action="store_true", help="Test servers after starting")
    
    args = parser.parse_args()
    
    # Create the model store directory if it doesn't exist
    os.makedirs("./model_store", exist_ok=True)
    
    # Start both servers
    threads = []
    
    try:
        # Start ModelServer
        model_server_thread = start_model_server(args.host, args.model_server_port)
        threads.append(model_server_thread)
        
        # Start SimpleAPI
        simple_api_thread = start_simple_api(args.host, args.simple_api_port)
        threads.append(simple_api_thread)
        
        # Test if requested
        if args.test:
            # Wait a bit for servers to fully initialize
            time.sleep(2)
            model_server_ok, simple_api_ok = test_servers(args.model_server_port, args.simple_api_port)
            
            if not model_server_ok and not simple_api_ok:
                logger.error("Both servers failed tests")
                sys.exit(1)
        
        # Wait for Ctrl+C
        print("\nServers are running. Press Ctrl+C to stop.\n")
        
        # Keep the main thread alive
        while True:
            if not all(t.is_alive() for t in threads):
                logger.error("One or more server threads have died")
                break
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    except Exception as e:
        logger.error(f"Error starting servers: {e}")
    
    print("Servers stopped.")

if __name__ == "__main__":
    main() 
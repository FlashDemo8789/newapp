#!/bin/bash

# Set environment variables
export FLASK_APP=backend/wsgi.py
export PYTHONPATH=$PWD
export FLASK_DNA_ROOT=$PWD

# Check if development mode is specified
if [ "$1" == "dev" ]; then
    export FLASK_ENV=development
    export FLASK_DEBUG=1
    export USE_MOCK_DATA=true
    echo "Starting API server in development mode..."
else
    export FLASK_ENV=production
    export FLASK_DEBUG=0
    export USE_MOCK_DATA=false
    echo "Starting API server in production mode..."
fi

# Check if port argument was provided
PORT=${2:-5000}
echo "Using port: $PORT"

# Run the Flask application
python3 -m backend.wsgi

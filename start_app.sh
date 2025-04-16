#!/bin/bash

# Start the FastAPI backend
echo "Starting FastAPI backend..."
cd backend
python -m pip install -r requirements.txt
cd app
python -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait a moment for the backend to initialize
sleep 2

# Start the React frontend
echo "Starting React frontend..."
cd ../../frontend
npm install
npm start &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Function to handle script termination
cleanup() {
  echo "Shutting down servers..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  exit 0
}

# Set up trap to catch termination signal
trap cleanup SIGINT

# Keep script running
echo "FlashDNA app is running. Press Ctrl+C to stop."
wait

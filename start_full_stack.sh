#!/bin/bash

echo "Starting AI Financial Assistant Full Stack..."
echo

echo "Starting Backend Server..."
cd backend
python setup_and_run.py &
BACKEND_PID=$!

echo "Waiting for backend to initialize..."
sleep 10

echo "Starting Frontend Development Server..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo
echo "Both servers are running:"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo
echo "Press Ctrl+C to stop both servers..."

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait

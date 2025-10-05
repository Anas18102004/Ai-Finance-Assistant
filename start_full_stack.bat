@echo off
echo Starting AI Financial Assistant Full Stack...
echo.

echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && python setup_and_run.py"

echo Waiting for backend to initialize...
timeout /t 10 /nobreak > nul

echo Starting Frontend Development Server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit...
pause > nul

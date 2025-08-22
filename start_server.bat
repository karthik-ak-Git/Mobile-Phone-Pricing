@echo off
echo.
echo ========================================
echo   Mobile Phone Price Predictor API
echo   Starting FastAPI Server...
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -q fastapi uvicorn python-multipart jinja2

REM Start the FastAPI server
echo.
echo Starting server at http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause

@echo off
echo Installing dependencies if needed...
pip install -r requirements.txt --quiet
echo.
echo Starting RoboPacerV2 Trainer...
echo Open your browser at http://localhost:5000
echo.
python "%~dp0engine\server.py"
pause

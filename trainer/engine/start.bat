@echo off
echo Installing dependencies if needed...
pip install -r "%~dp0..\requirements.txt" --quiet
echo.
echo Starting RoboPacerV2 Trainer...
echo Open your browser at http://localhost:5000
echo.
python "%~dp0server.py"
pause

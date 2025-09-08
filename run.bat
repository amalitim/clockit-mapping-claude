@echo off
echo Starting Task Type Classifier...
cd /d "%~dp0"
.venv\Scripts\activate
python app.py
pause
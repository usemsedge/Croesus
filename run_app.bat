@echo off
REM Run Flask app on Windows
REM Windows doesn't need DYLD_LIBRARY_PATH (that's macOS-specific)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the Flask app
python app.py

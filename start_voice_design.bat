@echo off
setlocal enabledelayedexpansion
title Qwen3-TTS VoiceDesign Studio
cd /d "%~dp0"

echo =====================================================================
echo           Qwen3-TTS 1.7B VoiceDesign Local Studio (RTX 4060)
echo =====================================================================
echo.

set "PYTHONUTF8=1"
if not defined HF_HOME (
    set "HF_HOME=%~dp0models_cache"
)

:: Verify Virtual Environment
if not exist "%~dp0venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at: "%~dp0venv"
    echo Please make sure the setup completed successfully.
    pause
    exit /b 1
)

call "%~dp0venv\Scripts\activate.bat"

echo [OK] Virtual Environment Activated.
echo [OK] Launching Professional Web Interface...
echo.
echo =====================================================================
echo Web Studio URL: http://127.0.0.1:7860
echo.
echo NOTE: Bind is strictly 127.0.0.1 (local machine only).
echo Keep this terminal window open while using the app.
echo Press Ctrl+C in this window to stop the server.
echo =====================================================================
echo.

python "%~dp0app.py"

echo.
echo Server has stopped.
pause

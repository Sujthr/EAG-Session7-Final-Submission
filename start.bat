@echo off
title S7 RAG Agent Launcher
echo.
echo  Starting S7 RAG Agent...
echo  (A PowerShell window will open with progress)
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
    echo.
    echo  start.ps1 exited with an error. See above.
    pause
)

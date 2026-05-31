@echo off
title S7 RAG Agent — Stop
color 0C

echo.
echo  ==========================================
echo   S7 RAG Agent  ^|  Stopping...
echo  ==========================================
echo.

set FOUND=0

REM ── Kill Gateway (port 8107) ──────────────────────────────────────────────────
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":8107 "') do (
    if not "%%p"=="0" if not "%%p"=="" (
        echo  Stopping Gateway  [PID %%p] ...
        taskkill /F /PID %%p >nul 2>&1
        set FOUND=1
    )
)

REM ── Kill Streamlit (port 8501) ────────────────────────────────────────────────
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":8501 "') do (
    if not "%%p"=="0" if not "%%p"=="" (
        echo  Stopping Streamlit [PID %%p] ...
        taskkill /F /PID %%p >nul 2>&1
        set FOUND=1
    )
)

echo.
if "%FOUND%"=="0" (
    echo  No running Gateway or Streamlit processes found.
) else (
    echo  Done — all processes stopped.
)
echo.
timeout /t 2 /nobreak >nul

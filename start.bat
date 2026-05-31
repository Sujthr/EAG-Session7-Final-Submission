@echo off
title S7 RAG Agent Launcher
color 0A

echo.
echo  ==========================================
echo   S7 RAG Agent  ^|  OpenAI Edition
echo  ==========================================
echo.

REM ── Prerequisites ────────────────────────────────────────────────────────────
if not exist "%~dp0Gateway\.env" (
    echo  [ERROR] Gateway\.env not found!
    echo.
    echo  Create it first:
    echo    copy "%~dp0Gateway\.env.example" "%~dp0Gateway\.env"
    echo  Then open it and set OPENAI_API_KEY=sk-...
    echo.
    pause
    exit /b 1
)

where uv >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] uv not found. Install with:  pip install uv
    echo.
    pause
    exit /b 1
)

REM ── Clear stale processes ─────────────────────────────────────────────────────
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /R ":8107 "') do (
    if not "%%p"=="0" if not "%%p"=="" (
        echo  Clearing port 8107 [PID %%p]...
        taskkill /F /PID %%p >nul 2>&1
    )
)

REM ── [1/3] Gateway ─────────────────────────────────────────────────────────────
echo  [1/3] Starting LLM Gateway V7...
start "LLM Gateway V7" "%~dp0Gateway\run_gateway.bat"

REM Wait up to 90 seconds for gateway to be ready
echo  Waiting for Gateway to respond on port 8107
set /a TRIES=0
:wait_loop
    set /a TRIES+=1
    if %TRIES% GTR 45 (
        echo.
        echo  [ERROR] Gateway did not start within 90s.
        echo  Check the "LLM Gateway V7" window for errors.
        pause
        exit /b 1
    )
    timeout /t 2 /nobreak >nul
    curl -sf http://localhost:8107/v1/status >nul 2>&1
    if errorlevel 1 (
        set /p =.< nul
        goto wait_loop
    )

echo.
echo  [OK] Gateway ready at http://localhost:8107

REM ── [2/3] Streamlit ───────────────────────────────────────────────────────────
echo  [2/3] Starting Streamlit dashboard...
start "Streamlit Dashboard" "%~dp0S7Code\run_streamlit.bat"
timeout /t 8 /nobreak >nul
start "" http://localhost:8501
echo  [OK] Dashboard at http://localhost:8501

REM ── [3/3] Query runner ────────────────────────────────────────────────────────
echo  [3/3] Launching benchmark query runner...
start "Query Runner" "%~dp0S7Code\run_queries.bat"
echo  [OK] Query Runner launched

echo.
echo  ==========================================
echo   ALL SYSTEMS RUNNING
echo    Gateway  : http://localhost:8107
echo    Dashboard: http://localhost:8501
echo    Queries  : see "Query Runner" window
echo  ==========================================
echo.
echo  Press any key to close this launcher.
pause >nul

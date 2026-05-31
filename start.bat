@echo off
title S7 RAG Agent — OpenAI Edition
color 0A

echo.
echo  ==========================================
echo   S7 RAG Agent  ^|  OpenAI Edition
echo  ==========================================
echo.

REM ── Store root (no trailing backslash) ────────────────────────────────────────
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM ── Prerequisites ─────────────────────────────────────────────────────────────
if not exist "%ROOT%\Gateway\.env" (
    echo  [ERROR] Gateway\.env not found!
    echo.
    echo  Please create it:
    echo    copy "%ROOT%\Gateway\.env.example" "%ROOT%\Gateway\.env"
    echo  Then open it and set OPENAI_API_KEY=sk-...
    echo.
    pause
    exit /b 1
)

where uv >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] uv not found. Install it first:
    echo    pip install uv
    echo.
    pause
    exit /b 1
)

where curl >nul 2>&1
if errorlevel 1 (
    set NO_CURL=1
) else (
    set NO_CURL=0
)

REM ── Gateway ────────────────────────────────────────────────────────────────────
echo  [1/3] Starting Gateway V7 on port 8107 ...
start "LLM Gateway V7" /D "%ROOT%\Gateway\llm_gatewayV7" cmd /k "title LLM Gateway V7 && uv run main.py"

if "%NO_CURL%"=="0" (
    echo  Waiting for Gateway to be ready ...
    set TRIES=0
    :wait_loop
        set /a TRIES+=1
        if %TRIES% GTR 60 (
            echo  [ERROR] Gateway did not respond within 60s. Check the Gateway window.
            pause
            exit /b 1
        )
        timeout /t 1 /nobreak >nul
        curl -sf http://localhost:8107/v1/status >nul 2>&1
        if errorlevel 1 goto wait_loop
    echo  [OK] Gateway ready at http://localhost:8107
) else (
    echo  Waiting 8s for Gateway to start (curl not available to verify)...
    timeout /t 8 /nobreak >nul
    echo  [OK] Gateway started
)

REM ── Streamlit ──────────────────────────────────────────────────────────────────
echo  [2/3] Starting Streamlit dashboard on port 8501 ...
start "Streamlit Dashboard" /D "%ROOT%\S7Code\S7code" cmd /k "title Streamlit Dashboard && uv run streamlit run app.py"
timeout /t 5 /nobreak >nul
start "" http://localhost:8501

REM ── Run all queries ────────────────────────────────────────────────────────────
echo  [3/3] Launching all benchmark queries (A-H + Q1-Q5) ...
echo  (watch the Query Runner window for live output)
echo.
start "Query Runner" /D "%ROOT%\S7Code\S7code" cmd /k "title Query Runner && uv run python run_all_queries.py"

echo.
echo  ==========================================
echo   ALL SYSTEMS RUNNING
echo    Gateway   : http://localhost:8107
echo    Dashboard : http://localhost:8501
echo    Queries   : running in Query Runner window
echo  ==========================================
echo.
echo  Press any key to close this launcher window.
pause >nul

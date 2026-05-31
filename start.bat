@echo off
title S7 RAG Agent — OpenAI Edition
color 0A

echo.
echo  ==========================================
echo   S7 RAG Agent  ^|  OpenAI Edition
echo  ==========================================
echo.

REM ── Prerequisites ─────────────────────────────────────────────────────────────
if not exist "%~dp0Gateway\.env" (
    echo  [ERROR] Gateway\.env not found!
    echo.
    echo  Please create it:
    echo    copy "%~dp0Gateway\.env.example" "%~dp0Gateway\.env"
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
    echo  [WARN] curl not found — gateway readiness check will be skipped.
    set NO_CURL=1
) else (
    set NO_CURL=0
)

REM ── Gateway ────────────────────────────────────────────────────────────────────
echo  [1/2] Starting Gateway V7 on port 8107 ...
start "LLM Gateway V7" cmd /k "title LLM Gateway V7 ^& cd /d "%~dp0Gateway\llm_gatewayV7" ^& uv run main.py"

if "%NO_CURL%"=="0" (
    echo  Waiting for Gateway ...
    set TRIES=0
    :wait_loop
        set /a TRIES+=1
        if %TRIES% GTR 60 (
            echo  [ERROR] Gateway did not start within 60 s. Check the Gateway window.
            pause
            exit /b 1
        )
        timeout /t 1 /nobreak >nul
        curl -sf http://localhost:8107/v1/status >nul 2>&1
        if errorlevel 1 goto wait_loop
    echo  [OK] Gateway  http://localhost:8107
) else (
    timeout /t 6 /nobreak >nul
    echo  [OK] Gateway started (curl not available to verify)
)

REM ── Streamlit ──────────────────────────────────────────────────────────────────
echo  [2/2] Starting Streamlit dashboard on port 8501 ...
start "Streamlit Dashboard" cmd /k "title Streamlit Dashboard ^& cd /d "%~dp0S7Code\S7code" ^& uv run streamlit run app.py"
timeout /t 4 /nobreak >nul
start http://localhost:8501

echo.
echo  ==========================================
echo   RUNNING
echo    Gateway   : http://localhost:8107
echo    Dashboard : http://localhost:8501
echo  ==========================================
echo.
echo  To run a query, open a new terminal and type:
echo    cd "%~dp0S7Code\S7code"
echo    uv run agent7.py "your query here"
echo.
echo  Press any key to close this window.
pause >nul

@echo off
title LLM Gateway V7
cd /d "%~dp0llm_gatewayV7"
echo  Starting LLM Gateway V7...
echo  Port: 8107
echo.
uv run main.py
echo.
echo  Gateway stopped. Press any key to close.
pause >nul

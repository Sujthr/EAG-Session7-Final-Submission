@echo off
title Query Runner
cd /d "%~dp0S7code"
echo  Running all benchmark queries (A-H + Q1-Q5)...
echo  Logs will be saved to: logs\
echo.
uv run python run_all_queries.py
echo.
echo  All queries done. Press any key to close.
pause >nul

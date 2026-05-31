@echo off
title Streamlit Dashboard
cd /d "%~dp0S7code"
echo  Starting Streamlit Dashboard...
echo  URL: http://localhost:8501
echo.
uv run streamlit run app.py
echo.
echo  Streamlit stopped. Press any key to close.
pause >nul

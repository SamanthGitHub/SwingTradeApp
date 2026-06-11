@echo off
REM SwingTrade Pro -- self-contained Windows launcher.
REM Creates an isolated Python virtual environment in your user profile,
REM installs the pinned dependencies once, then runs the Streamlit dashboard.
REM
REM Usage:  double-click run.bat in Explorer, or run  run.bat  in a terminal.
REM         set SETUP_ONLY=1 to build the venv without launching.
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM -- 1. Find a usable Python (prefer the py launcher; avoid the too-new 3.14) ----
set "PYCMD="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PYCMD (
    py -%%V -c "import sys" >nul 2>&1 && set "PYCMD=py -%%V"
  )
)
if not defined PYCMD (
  py -3 -c "import sys" >nul 2>&1 && set "PYCMD=py -3"
)
if not defined PYCMD (
  python -c "import sys" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
  echo.
  echo  No Python found. Install Python 3.12 from https://www.python.org/downloads/
  echo  During setup, tick "Add Python to PATH", then run this file again.
  echo.
  pause
  exit /b 1
)
echo Using Python: !PYCMD!
!PYCMD! --version

REM -- 2. venv lives in your user profile (not in the synced Drive folder) ---------
set "VENV=%USERPROFILE%\.swingtradeapp\venv"
if not exist "%VENV%\Scripts\python.exe" (
  echo Creating isolated virtual environment at %VENV% ...
  !PYCMD! -m venv "%VENV%"
)
set "VPY=%VENV%\Scripts\python.exe"

REM -- 3. Install/refresh deps only when requirements.txt changes ------------------
"%VPY%" -m pip install --quiet --upgrade pip
set "REQHASH="
for /f "skip=1 tokens=* delims=" %%H in ('certutil -hashfile requirements.txt SHA1') do (
  if not defined REQHASH set "REQHASH=%%H"
)
set "REQHASH=!REQHASH: =!"
set "STAMP=%VENV%\.req-!REQHASH!"
if not exist "!STAMP!" (
  echo Installing dependencies ^(first run or requirements changed^)...
  "%VPY%" -m pip install -r requirements.txt
  del /q "%VENV%\.req-*" >nul 2>&1
  type nul > "!STAMP!"
)

REM -- 4. Launch (or just verify, under SETUP_ONLY) -------------------------------
if "%SETUP_ONLY%"=="1" (
  "%VPY%" -c "import streamlit, plotly, yfinance, pandas, numpy; print('core imports OK')"
  echo Setup complete. venv: %VENV%
  exit /b 0
)
echo.
echo Launching SwingTrade Pro -- http://localhost:8501  (close this window to stop)
echo.
"%VPY%" -m streamlit run app.py

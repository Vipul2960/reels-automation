@echo off
setlocal
title Reels Maker
cd /d "%~dp0"

echo.
echo   ================================================
echo     REELS MAKER
echo   ================================================
echo.
echo   Paste a YouTube link, or drag a video file here.
echo.

set "TARGET="
set /p TARGET=  Link or file: 
if "%TARGET%"=="" (
  echo.
  echo   Nothing entered. Closing.
  pause
  exit /b 1
)
rem Drag-and-drop wraps the path in quotes; strip them.
set TARGET=%TARGET:"=%

set "COUNT=12"
set /p COUNT=  How many reels [12]: 

echo.
echo   Working. Leave this window open - progress shows below.
echo   ------------------------------------------------
echo.

venv\Scripts\python.exe run.py "%TARGET%" --clips %COUNT%
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo   ------------------------------------------------
  echo   Done. Reels are in the output folder.
  echo   ------------------------------------------------
  start "" "%~dp0output"
) else (
  echo   ------------------------------------------------
  echo   Stopped with an error ^(code %CODE%^). The message
  echo   above says what went wrong.
  echo   ------------------------------------------------
)
echo.
pause

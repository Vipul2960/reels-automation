@echo off
setlocal enabledelayedexpansion
title Reels Maker
cd /d "%~dp0"

echo.
echo   ================================================
echo     REELS MAKER
echo   ================================================
echo.
echo     1  One link
echo     2  Many links  (overnight batch)
echo.

set "MODE=1"
set /p MODE=  Choose [1]:

set "TARGET="
if "%MODE%"=="2" goto OPTIONS

echo.
set /p TARGET=  Paste the link:
if "%TARGET%"=="" (
  echo.
  echo   Nothing entered. Closing.
  pause
  exit /b 1
)
set TARGET=%TARGET:"=%

:OPTIONS
echo.
set "COUNT=12"
set /p COUNT=  How many reels per video [12]:

set "UPFLAG="
if not "%MODE%"=="2" (
  set "UP="
  set /p UP=  Ask to upload each one to YouTube? [y/N]:
  if /i "!UP!"=="y"   set "UPFLAG=--upload"
  if /i "!UP!"=="yes" set "UPFLAG=--upload"
)

set "SKIPFLAG="
set "CLEANFLAG="
if "%MODE%"=="2" (
  echo.
  echo   For an unattended run, these stop it waiting on questions:
  set "SK="
  set /p SK=  Skip videos already done? [Y/n]:
  if /i not "!SK!"=="n" set "SKIPFLAG=--skip-existing"

  set "CL="
  set /p CL=  Delete each source video after its reels are made? [Y/n]:
  if /i not "!CL!"=="n" set "CLEANFLAG=--clean"
)

echo.
echo   ------------------------------------------------

if "%MODE%"=="2" (
  rem No link on the command line, so run.py asks for them itself -
  rem paste as many as you like, then press Enter on a blank line.
  venv\Scripts\python.exe run.py --clips %COUNT% %SKIPFLAG% %CLEANFLAG%
) else (
  venv\Scripts\python.exe run.py "%TARGET%" --clips %COUNT% %UPFLAG%
)
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo   ------------------------------------------------
  echo   Done. Reels are in the output folder.
  if "%MODE%"=="2" echo   Read output\batch-summary.txt for what happened.
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

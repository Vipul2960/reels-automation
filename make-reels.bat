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
set "LINKFLAG="

if "%MODE%"=="2" goto MANY

rem ---------------------------------------------------------------- one link
echo.
set /p TARGET=  Paste the link:
if "%TARGET%"=="" (
  echo.
  echo   Nothing entered. Closing.
  pause
  exit /b 1
)
set TARGET=%TARGET:"=%
goto OPTIONS

rem --------------------------------------------------------------- many links
:MANY
if not exist links.txt (
  echo # One YouTube link per line. Lines starting with # are ignored.> links.txt
  echo.>> links.txt
)
echo.
echo   Opening links.txt - paste your links, one per line, then SAVE and CLOSE it.
echo.
start /wait notepad.exe links.txt
set "LINKFLAG=--links links.txt"

set "N=0"
for /f "usebackq tokens=* delims=" %%L in ("links.txt") do (
  set "LINE=%%L"
  if not "!LINE!"=="" if not "!LINE:~0,1!"=="#" set /a N+=1
)
echo   %N% link^(s^) found.
if "%N%"=="0" (
  echo   Nothing to do. Closing.
  pause
  exit /b 1
)

rem ------------------------------------------------------------------ options
:OPTIONS
echo.
set "COUNT=12"
set /p COUNT=  How many reels per video [12]:

set "UP="
set /p UP=  Ask to upload each one to YouTube? [y/N]:
set "UPFLAG="
if /i "%UP%"=="y"   set "UPFLAG=--upload"
if /i "%UP%"=="yes" set "UPFLAG=--upload"

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
echo   Working. Leave this window open - progress shows below.
echo   ------------------------------------------------
echo.

if "%MODE%"=="2" (
  venv\Scripts\python.exe run.py %LINKFLAG% --clips %COUNT% %UPFLAG% %SKIPFLAG% %CLEANFLAG%
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

@echo off
cd /d "%~dp0"
title chesSon Chess Bot - Match Setup
cls
echo ================================================================
echo             chesSon Neural Chess Engine - Setup
echo ================================================================
echo.

set /p LEVEL="1. Choose Stockfish Level (1-8) [Default 1]: "
if "%LEVEL%"=="" set LEVEL=1

echo.
echo 2. Choose your Bot's Color:
echo    [1] White
echo    [2] Black
echo    [3] Random
set /p COLOR_CHOICE="Select (1-3) [Default 1]: "
if "%COLOR_CHOICE%"=="2" (
    set COLOR=black
) else if "%COLOR_CHOICE%"=="3" (
    set COLOR=random
) else (
    set COLOR=white
)

echo.
echo 3. Choose Time Control:
echo    [1] 1 min + 0s increment  (Bullet / Hyper-Rapid)
echo    [2] 1 min + 1s increment  (Bullet)
echo    [3] 3 min + 2s increment  (Blitz)
echo    [4] 5 min + 3s increment  (Blitz)
echo    [5] 10 min + 0s increment (Rapid)
echo    [6] 15 min + 10s increment (Rapid)
set /p TIME_CHOICE="Select (1-6) [Default 1]: "

if "%TIME_CHOICE%"=="1" (
    set TIME_SEC=60
    set INC_SEC=0
) else if "%TIME_CHOICE%"=="2" (
    set TIME_SEC=60
    set INC_SEC=1
) else if "%TIME_CHOICE%"=="3" (
    set TIME_SEC=180
    set INC_SEC=2
) else if "%TIME_CHOICE%"=="4" (
    set TIME_SEC=300
    set INC_SEC=3
) else if "%TIME_CHOICE%"=="5" (
    set TIME_SEC=600
    set INC_SEC=0
) else if "%TIME_CHOICE%"=="6" (
    set TIME_SEC=900
    set INC_SEC=10
) else (
    set TIME_SEC=60
    set INC_SEC=0
)

cls
echo ================================================================
echo Starting chesSon vs Stockfish Level %LEVEL%
echo Color: %COLOR%  ^|  Time: %TIME_SEC%s + %INC_SEC%s
echo ================================================================
echo.

python lichess_bot.py --token lip_63UwOWa79OR0uDvmCu5k --weights chesson\weights\chesson_weights.pt --level %LEVEL% --color %COLOR% --time %TIME_SEC% --increment %INC_SEC%

pause

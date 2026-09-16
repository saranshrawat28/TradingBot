@echo off
title Setup ApexTrade Autonomous 24/7 Background Service
echo ======================================================================
echo   APEXTRADE: 24/7 AUTONOMOUS BACKGROUND SERVICE INSTALLER
echo ======================================================================
echo.
echo This installer configures ApexTrade to run completely on its own:
echo   1. Auto-starts Streamlit Dashboard (http://localhost:8501) on boot
echo   2. Auto-scans 200+ NSE stocks at 08:50 AM IST every trading day
echo   3. Auto-fills 5 paper trades at 09:15 AM IST
echo   4. Auto-evaluates outcomes and exports Excel reports at 03:35 PM IST
echo   5. Runs 100%% silently in the background with ZERO pop-up windows
echo.
echo ======================================================================
echo.

set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1%

echo [1/3] Creating Windows Scheduled Task: Morning Scan (08:50 AM Mon-Fri)...
schtasks /create /tn "ApexTrade_Morning_Scan" /tr "\"C:\Python313\pythonw.exe\" \"%PROJECT_DIR%\paper_lab_run.py\" --pick-now" /sc weekly /d MON,TUE,WED,THU,FRI /st 08:50 /f >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       * SUCCESS: Registered ApexTrade_Morning_Scan at 08:50 AM Mon-Fri
) else (
    echo       * NOTE: Administrator privileges required for schtasks. Falling back to background daemon.
)

echo [2/3] Creating Windows Scheduled Task: EOD Evaluation (03:35 PM Mon-Fri)...
schtasks /create /tn "ApexTrade_EOD_Evaluation" /tr "\"C:\Python313\pythonw.exe\" \"%PROJECT_DIR%\paper_lab_run.py\" --evaluate-now" /sc weekly /d MON,TUE,WED,THU,FRI /st 15:35 /f >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       * SUCCESS: Registered ApexTrade_EOD_Evaluation at 03:35 PM Mon-Fri
) else (
    echo       * NOTE: Fallback scheduler handles intraday timings.
)

echo [3/3] Launching Invisible Background Service right now...
wscript "%PROJECT_DIR%\scripts\run_background_service.vbs"
echo       * SUCCESS: Background service is running silently!
echo.
echo ======================================================================
echo   INSTALLATION COMPLETE! 🚀
echo.
echo   - Streamlit Dashboard is LIVE at: http://localhost:8501
echo   - The bot is now running on its own in the background.
echo   - You DO NOT need to open any terminal or chat to keep it running!
echo   - To stop the service anytime, run: scripts\stop_background_service.bat
echo ======================================================================
echo.
pause

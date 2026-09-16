@echo off
title Stop ApexTrade Background Service
echo ========================================================
echo   Stopping ApexTrade Background Services...
echo ========================================================
echo.

taskkill /F /IM pythonw.exe 2>nul
taskkill /F /FI "WINDOWTITLE eq ApexTrade*" 2>nul

echo.
echo ApexTrade background processes stopped successfully.
pause

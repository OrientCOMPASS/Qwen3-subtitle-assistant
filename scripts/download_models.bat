@echo off
rem Downloads all required models into ..\models\
rem For users in China, prefer the PowerShell command with -HfMirror (see README):
rem   powershell -ExecutionPolicy Bypass -File download_models.ps1 -HfMirror https://hf-mirror.com
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_models.ps1" %*
echo.
pause

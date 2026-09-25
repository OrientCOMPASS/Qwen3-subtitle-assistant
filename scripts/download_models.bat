@echo off
rem 双击即可下载全部模型（国内网络建议改用带 -HfMirror 参数的 PowerShell 命令，见 README）
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_models.ps1" %*
echo.
pause

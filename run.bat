@echo off
setlocal
chcp 65001 > nul

pushd "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv is not installed.
    echo [ERROR] Install uv: pip install uv
    pause >nul
    popd
    exit /b 1
)

uv run main.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo Task finished. Press any key to exit...
pause >nul
popd
exit /b %EXIT_CODE%

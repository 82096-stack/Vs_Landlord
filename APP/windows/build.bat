@echo off
echo ========================================
echo  DouDiZhu - Windows PyInstaller Build
echo ========================================
echo.

REM Check if pyinstaller is installed
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Install project dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build with spec file
echo Building...
pyinstaller main.spec

echo.
echo ========================================
echo Build complete! Executable is in dist\DouDiZhu.exe
echo ========================================
pause

@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo.
    echo [SpectroProtz] No .venv found. Falling back to system Python.
    echo.
)
python -m spectro_app.main
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" (
    echo SpectroProtz exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%

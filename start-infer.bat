@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe app.py --infer
echo.
echo ==== exited, code: %errorlevel% (non-zero = see errors above) ====
pause

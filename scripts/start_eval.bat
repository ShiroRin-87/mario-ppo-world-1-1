@echo off
pushd "%~dp0.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\setup.ps1 first.
    popd
    exit /b 1
)
if "%~1"=="" (
    ".venv\Scripts\python.exe" eval\evaluate.py --model models\current\best_model.zip --episodes 10
) else (
    ".venv\Scripts\python.exe" eval\evaluate.py %*
)
set "result=%errorlevel%"
popd
exit /b %result%

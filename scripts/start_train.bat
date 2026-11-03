@echo off
pushd "%~dp0.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\setup.ps1 first.
    popd
    exit /b 1
)
if "%~1"=="" (
    ".venv\Scripts\python.exe" train\train_ppo.py --config configs\baseline.yaml
) else (
    ".venv\Scripts\python.exe" train\train_ppo.py %*
)
set "result=%errorlevel%"
popd
exit /b %result%

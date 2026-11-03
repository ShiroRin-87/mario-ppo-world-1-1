@echo off
pushd "%~dp0.." || exit /b 1
if not exist "data\demo.mp4" (
    call scripts\start_eval.bat --model models\current\best_model.zip --episodes 1 --video data\demo.mp4
    if errorlevel 1 (
        popd
        exit /b 1
    )
)
start "" "data\demo.mp4"
popd

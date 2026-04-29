@echo off
setlocal
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" infer.py --image-dir data\images --output-dir outputs\inference_run

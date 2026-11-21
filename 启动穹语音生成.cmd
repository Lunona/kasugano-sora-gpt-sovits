@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not defined SORAVOICE_PYTHON set "SORAVOICE_PYTHON=F:\GPT-SoVITS\venv\Scripts\python.exe"
"%SORAVOICE_PYTHON%" -X utf8 "%~dp0webui_sora.py" --open %*
pause

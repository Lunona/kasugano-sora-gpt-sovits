@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
"F:\GPT-SoVITS\venv\Scripts\python.exe" -X utf8 "%~dp0webui_sora.py" --open %*
pause

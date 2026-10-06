@echo off
rem Cria o ambiente na primeira vez e inicia o app. Uso: run.bat [--port 8765] [--no-browser]
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Criando ambiente Python (.venv)...
  py -3 -m venv .venv || python -m venv .venv || goto :fail
  ".venv\Scripts\python.exe" -m pip install -e . || goto :fail
)
".venv\Scripts\python.exe" -m smart_photo_edit %*
exit /b %errorlevel%
:fail
echo Falha ao preparar o ambiente. Instale o Python 3.10+ em python.org e tente de novo.
exit /b 1

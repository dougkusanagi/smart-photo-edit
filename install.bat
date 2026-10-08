@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto :dependencies
where py >nul 2>nul
if errorlevel 1 goto :python
py -3 -m venv .venv
if errorlevel 1 goto :fail
goto :dependencies
:python
python -m venv .venv
if errorlevel 1 goto :fail
:dependencies
".venv\Scripts\python.exe" -m pip --version >nul 2>nul
if not errorlevel 1 goto :pip
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto :fail
:pip
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :fail
echo Instalacao concluida. Execute run.bat para abrir o aplicativo.
exit /b 0
:fail
echo Falha na instalacao. Confira a mensagem acima, Python 3.12/3.13 com venv e a conexao.
pause
exit /b 1

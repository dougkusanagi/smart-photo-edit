@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto :venv
where py >nul 2>nul
if errorlevel 1 goto :python
py -3 scripts\update_app.py
if errorlevel 1 goto :fail
goto :install
:python
python scripts\update_app.py
if errorlevel 1 goto :fail
goto :install
:venv
".venv\Scripts\python.exe" scripts\update_app.py
if errorlevel 1 goto :fail
:install
call install.bat
exit /b %errorlevel%
:fail
echo Falha ao atualizar. Confira a mensagem acima; a instalacao nao foi iniciada.
pause
exit /b 1

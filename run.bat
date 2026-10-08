@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :install
".venv\Scripts\python.exe" -c "import aiohttp, PIL, smart_photo_edit" >nul 2>nul
if not errorlevel 1 goto :run
:install
call install.bat
if errorlevel 1 exit /b 1
:run
".venv\Scripts\python.exe" -m smart_photo_edit %*
if errorlevel 1 goto :fail
exit /b 0
:fail
echo Falha ao iniciar. Confira a mensagem acima.
pause
exit /b 1

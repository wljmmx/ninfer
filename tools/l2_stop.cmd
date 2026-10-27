@echo off
setlocal
for /f "usebackq delims=" %%p in (`wsl wslpath -a "%~dp0l2_stop.sh"`) do set "SH=%%p"
if not defined SH (echo FAIL: wslpath failed & exit /b 1)
wsl -e bash "%SH%"
exit /b %errorlevel%

@echo off
rem Windows entry point for the L2 stack: converts this path and runs the shell script in WSL.
setlocal
for /f "usebackq delims=" %%p in (`wsl wslpath -a "%~dp0l2_start.sh"`) do set "SH=%%p"
if not defined SH (echo FAIL: wslpath failed & exit /b 1)
wsl -e bash "%SH%"
exit /b %errorlevel%

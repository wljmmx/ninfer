@echo off
rem One-click LMCache for the console. LMCache lives inside WSL (a venv, not a container), so this
rem script is the bridge: it maps itself into the WSL filesystem and runs l3_lmcache.sh there.
rem
rem   start_lmcache.cmd          start (idempotent) and report
rem   start_lmcache.cmd status   is it answering
rem   start_lmcache.cmd stop     stop the one it started
rem
rem The ports and the chunk size are a contract with the engine, not preferences:
rem   6555  ZMQ endpoint (the engine defaults to tcp://127.0.0.1:6555)
rem   18080 HTTP metrics (what this console's LMCache panel reads)
rem   128   chunk size: a checkpoint key is cut at whole 128-token chunks
setlocal
set "HERE=%~dp0"
set "SH=%HERE%l3_lmcache.sh"
if not exist "%SH%" (
  echo missing %SH%
  exit /b 3
)

rem wslpath is the only reliable bridge from a Windows path to the one bash sees.
set "SHP="
for /f "usebackq delims=" %%p in (`wsl.exe -e wslpath -a "%SH%"`) do set "SHP=%%p"
if "%SHP%"=="" (
  echo cannot map %SH% into WSL; is the WSL distro running?
  exit /b 3
)

wsl.exe -e bash -lc "bash '%SHP%' %*"
exit /b %ERRORLEVEL%

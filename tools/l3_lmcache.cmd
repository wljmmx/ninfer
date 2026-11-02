@echo off
rem Bring up the L3 tier for a Windows-side run: start the LMCache server inside WSL, then print the
rem two variables the engine needs. Nothing is installed and nothing runs in a container - lmcache
rem lives in a WSL venv (see tools/l3_lmcache.sh) and the engine reaches it over the WSL2 loopback.
rem
rem   tools\l3_lmcache.cmd          start (idempotent) and print the engine side
rem   tools\l3_lmcache.cmd status   is it answering
rem   tools\l3_lmcache.cmd stop     stop it
setlocal
set "PORT=%L3_PORT%"
if "%PORT%"=="" set "PORT=6555"

if /i "%~1"=="status" (
  wsl -e bash -lc "ss -ltn 2>/dev/null | grep -q ':%PORT% ' && echo up on %PORT% || echo not running"
  goto :eof
)
if /i "%~1"=="stop" (
  wsl -e bash -lc "pkill -f 'lmcache server --transport zmq --port %PORT%' && echo stopped || echo was not running"
  goto :eof
)

wsl -e bash -lc "cd /mnt/h 2>/dev/null; bash '%CD%\l3_lmcache.sh'"
if errorlevel 1 (
  echo FAILED: the WSL side did not come up
  exit /b 1
)

echo.
echo Engine side (set these in the serve launcher, not here):
echo   set NINFER_LMCACHE_ENDPOINT=tcp://127.0.0.1:%PORT%
echo   set NINFER_LMCACHE_RESTORE=1
echo.
echo The endpoint must be a ZMQ URL - "127.0.0.1:%PORT%" alone fails with "Invalid argument".
echo The server must keep --chunk-size 128: a checkpoint key is cut at whole 128-token chunks.
endlocal

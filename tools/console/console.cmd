@echo off
rem Serve the console and proxy LMCache's Prometheus metrics onto the same origin.
rem
rem Opening the html directly (file://) cannot work: LMCache sends no CORS headers, so the browser
rem refuses the cross-origin read and the panel would sit empty. Going through this server keeps
rem everything on http://127.0.0.1:8099 and the panel fills in.
setlocal
set "PORT=8099"
set "HERE=%~dp0"
rem Engine base URL: override with the first argument, e.g. console.cmd http://127.0.0.1:11438
set "ENGINE=%~1"
if "%ENGINE%"=="" set "ENGINE=http://127.0.0.1:11438"

rem "python" is not guaranteed on Windows: a bare `python` is often the Store stub, which only
rem opens the Store and exits. Resolve an interpreter that actually runs, in the order Windows
rem ships them, and say so plainly when none is found.
set "PY="
py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
if not defined PY ( python -c "import sys" >nul 2>&1 && set "PY=python" )
if not defined PY ( python3 -c "import sys" >nul 2>&1 && set "PY=python3" )
if not defined PY (
  echo No working Python 3 found.
  echo Install Python 3.10+ from python.org, or put py/python on PATH, then run this again.
  pause
  exit /b 3
)

rem LAN mode: the page and the engine it starts listen on every interface. The engine banner says
rem `auth disabled`, so this is only safe on a network you trust. Set LAN=0 to stay on 127.0.0.1.
set "LAN=1"
set "HOST=127.0.0.1"
set "EHOST=127.0.0.1"
if not "%LAN%"=="0" set "HOST=0.0.0.0"
if not "%LAN%"=="0" set "EHOST=0.0.0.0"

rem LMCache endpoint handed to an engine the page starts. The mirror-network address is the fast one
rem on this box; 127.0.0.1 also works but every byte goes through the user-mode relay, which is what
rem a 2.5 GB prefix restore notices. Set it before this script to use another server.
if not defined NINFER_LMCACHE_ENDPOINT set "NINFER_LMCACHE_ENDPOINT=tcp://10.222.222.1:6555"

echo console : http://%HOST%:%PORT%/   (engine on %EHOST%)
echo metrics : http://%HOST%:%PORT%/api/lmcache.json
echo gpu     : http://%HOST%:%PORT%/api/gpu.json
echo engine  : http://%HOST%:%PORT%/api/engine.json  (%ENGINE%)
echo python  : %PY%
echo.
echo Requires the LMCache server on http://127.0.0.1:18080/metrics (optional).
echo Pick a .ninfer file in the page to start the engine; nothing is shown until it answers.
echo Models root defaults to R:\model; add --models-root DIR to change it.
echo Press Ctrl+C to stop.
echo.
rem The engine talks to LMCache through the mirror-network address: 127.0.0.1 also works, but it
rem goes through the user-mode relay, which is measurably slower for a multi-gigabyte prefix.
if not defined NINFER_LMCACHE_ENDPOINT set "NINFER_LMCACHE_ENDPOINT=tcp://10.222.222.1:6555"
%PY% "%HERE%lmcache_console.py" --port %PORT% --engine %ENGINE% --host %HOST% --engine-host %EHOST% --lmcache-endpoint "%NINFER_LMCACHE_ENDPOINT%"
if errorlevel 1 (
  echo.
  echo The console exited with %ERRORLEVEL%.
  pause
)
endlocal

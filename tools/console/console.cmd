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

echo console : http://127.0.0.1:%PORT%/
echo metrics : http://127.0.0.1:%PORT%/api/lmcache.json
echo gpu     : http://127.0.0.1:%PORT%/api/gpu.json
echo engine  : http://127.0.0.1:%PORT%/api/engine.json  (%ENGINE%)
echo python  : %PY%
echo.
echo Requires the LMCache server on http://127.0.0.1:18080/metrics (optional).
echo Pick a .ninfer file in the page to start the engine; nothing is shown until it answers.
echo Models root defaults to R:\model; add --models-root DIR to change it.
echo Press Ctrl+C to stop.
echo.
%PY% "%HERE%lmcache_console.py" --port %PORT% --engine %ENGINE%
if errorlevel 1 (
  echo.
  echo The console exited with %ERRORLEVEL%.
  pause
)
endlocal

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
echo console : http://127.0.0.1:%PORT%/
echo metrics : http://127.0.0.1:%PORT%/api/lmcache.json
echo gpu     : http://127.0.0.1:%PORT%/api/gpu.json
echo engine  : http://127.0.0.1:%PORT%/api/engine.json  (%ENGINE%)
echo.
echo Requires the LMCache server on http://127.0.0.1:18080/metrics (default).
echo Engine source is the ninfer serve base above; without it the console says 演示数据.
echo Press Ctrl+C to stop.
echo.
python "%HERE%lmcache_console.py" --port %PORT% --engine %ENGINE%
if errorlevel 1 (
  echo.
  echo Failed. Is python on PATH? Try: py -3 "%HERE%lmcache_console.py" --port %PORT% --engine %ENGINE%
  pause
)
endlocal

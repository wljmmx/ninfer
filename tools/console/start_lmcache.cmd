@echo off
rem One-click LMCache launcher for the console.
rem
rem The console's LMCache panel reads http://127.0.0.1:18080/metrics. This script is what the
rem "start LMCache" button runs, so the command lives here, in one place, instead of in the page.
rem
rem Adjust the COMMAND line if your LMCache install uses a different entry point; everything else
rem (log, pid, stop) is handled by the console.
setlocal
set "PORT=18080"

rem --- the entry point -------------------------------------------------------
where lmcache_server >nul 2>&1
if not errorlevel 1 (
    set "CMD=lmcache_server --host 127.0.0.1 --port %PORT%"
    goto run
)

python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('lmcache') else 1)" >nul 2>&1
if not errorlevel 1 (
    set "CMD=python -m lmcache.server --host 127.0.0.1 --port %PORT%"
    goto run
)

echo LMCache is not installed on this machine (no lmcache_server on PATH, no python module "lmcache").
echo Install it, or edit the COMMAND section of %~nx0 to point at your entry point.
exit /b 3

:run
echo starting: %CMD%
%CMD%
exit /b %ERRORLEVEL%

@echo off
call "H:\vs_ide\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=H:\vs_ide\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;H:\vs_ide\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cmake --build "H:\TeleAgentWorks\ninfer-4090-port\ninfer\build-win" > "H:\TeleAgentWorks\ninfer-4090-port\ninfer\.temp\build_rkfix.log" 2>&1

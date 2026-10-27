@echo off
echo Registering profiler hook...
start /wait regsvr32 /s nprof.hook.dll
if errorlevel 1 ( 
	rem Re-register, but don't be silent this time
	regsvr32 nprof.hook.dll 
)
pause

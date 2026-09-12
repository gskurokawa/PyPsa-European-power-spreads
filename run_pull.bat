@echo off
REM Runs the ENTSO-E pull until it completes, restarting if it dies.
REM Every chunk is cached, so each restart resumes where the last stopped
REM rather than starting over. Safe to leave running overnight.
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat

set /a ATTEMPT=0

:loop
set /a ATTEMPT+=1
echo.
echo ================ attempt %ATTEMPT% at %TIME% ================
python scripts\pipeline\01_pull_entsoe.py
if %ERRORLEVEL%==0 goto done
if %ATTEMPT% GEQ 25 goto giveup
echo.
echo Exited with code %ERRORLEVEL%. Restarting in 60 seconds...
timeout /t 60 /nobreak >nul
goto loop

:done
echo.
echo Pull finished cleanly after %ATTEMPT% attempt^(s^).
goto end

:giveup
echo.
echo Gave up after %ATTEMPT% attempts. See logs\pull.log.

:end
endlocal

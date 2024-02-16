@echo off
setlocal

set ENV_NAME=BardoGPT
set ENV_YML=environment.yml

:: Esporta l'intero ambiente (conda + pip) in environment.yml
echo Esportazione dell'ambiente "%ENV_NAME%" in "%ENV_YML%"...
CALL conda env export --name %ENV_NAME% | findstr /v "prefix: " > %ENV_YML%
if errorlevel 1 (
    echo Errore durante l'esportazione dell'ambiente.
    exit /b %errorlevel%
)

:End
echo Esportazione completata.
endlocal

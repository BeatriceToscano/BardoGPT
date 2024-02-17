@echo off
:: Verifica se è amministratore
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Programma avviato con privilegi da amministratore.
) else (
    echo Richiesta privilegi da amministratore...
    :: Esegue lo script stesso con privilegi da amministratore
    powershell -Command "Start-Process cmd -ArgumentList '/c %~f0' -Verb runAs" >nul 2>&1
    exit /b
)
setlocal enableextensions enabledelayedexpansion

set ENV_NAME=BardoGPT
CALL conda init cmd.exe
:: Ottiene il percorso dell'ambiente base di Conda
for /f "tokens=*" %%i in ('conda info --base') do set BASE_ENV_PATH=%%i\envs

echo Ambienti Conda esistenti e le loro posizioni:
CALL conda info --envs

echo.
echo Inserisci il percorso dove vuoi creare l'ambiente %ENV_NAME%.
echo Digita 'default' per usare la posizione predefinita nella directory degli ambienti Conda: !BASE_ENV_PATH!
set /p ENV_PATH="Percorso ambiente (o 'd'): "

:: Gestisce la scelta dell'utente per la posizione dell'ambiente
if "%ENV_PATH%"=="d" (
    set PREFIX_CMD=--prefix "!BASE_ENV_PATH!\%ENV_NAME%"
) else (
    set PREFIX_CMD=--prefix "%ENV_PATH%\%ENV_NAME%"
)

:: Verifica se l'ambiente esiste
CALL conda info --envs | findstr /C:"%ENV_NAME%"
if errorlevel 1 (
    echo L'ambiente "%ENV_NAME%" non esiste. Creazione in corso...
    CALL conda create %PREFIX_CMD%
    if errorlevel 1 exit /b %errorlevel%
) else (
    echo L'ambiente "%ENV_NAME%" esiste già. Controllo e aggiornamento dei pacchetti in corso...

)


:: Attivazione dell'ambiente
CALL conda activate %ENV_NAME%
if errorlevel 1 exit /b %errorlevel%
:: conda
CALL conda install numpy -y
CALL conda install pandas -y
CALL conda install pytorch torchvision torchaudio cuda-toolkit=12.1 -c pytorch -c nvidia -y
CALL conda install tqdm -y
:: pip
CALL pip install mido[ports-rtmidi] -y
CALL pip install rich -y

CALL conda clean --all -y
echo Setup completato con successo.
endlocal

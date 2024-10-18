@echo off
setlocal

rem Get the directory of the current batch file
set "directory=%~dp0"

rem Change to the directory of the batch file
cd /d "%directory%"

rem Iterate through each .Spe file in the directory
for %%f in (*.Spe) do (
    echo Processing file: %%f
    rem Replace the next line with the command you want to run on each file
    python gaussian_dist_script.py "%%f"
)

endlocal
pause

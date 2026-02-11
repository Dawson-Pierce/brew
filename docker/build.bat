@echo off
setlocal

set SCRIPT_DIR=%~dp0
set CPP_DIR=%SCRIPT_DIR%..

echo Building BREW C++ in Docker...
docker build -t brew-cpp -f "%SCRIPT_DIR%Dockerfile" "%CPP_DIR%"

echo.
echo Running tests...
docker run --rm brew-cpp

endlocal

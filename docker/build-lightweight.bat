@echo off
setlocal

set SCRIPT_DIR=%~dp0
set CPP_DIR=%SCRIPT_DIR%..

echo Building BREW hardware image (plotting disabled)...
docker build -t brew-lw ^
  --build-arg BREW_ENABLE_PLOTTING=OFF ^
  --build-arg BREW_BUILD_TESTS=OFF ^
  -f "%SCRIPT_DIR%Dockerfile" "%CPP_DIR%"

echo.
echo Image ready: brew-hw

endlocal

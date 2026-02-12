@echo off
setlocal

set SCRIPT_DIR=%~dp0
set CPP_DIR=%SCRIPT_DIR%..

echo Building BREW dev image (plotting enabled)...
docker build -t brew-dev ^
  --build-arg BREW_ENABLE_PLOTTING=ON ^
  --build-arg BREW_BUILD_TESTS=ON ^
  -f "%SCRIPT_DIR%Dockerfile" "%CPP_DIR%"

echo.
echo Running tests (including plotting)...
docker run --rm -v "%CPP_DIR%\\output:/output" brew-dev

endlocal

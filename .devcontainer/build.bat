@echo off
setlocal

set SCRIPT_DIR=%~dp0
set CPP_DIR=%SCRIPT_DIR%..

echo Building BREW dev image...
docker build -t brew-dev ^
  --build-arg BREW_ENABLE_PLOTTING=ON ^
  -f "%SCRIPT_DIR%Dockerfile" "%CPP_DIR%"

if %ERRORLEVEL% neq 0 (
    echo Docker build failed.
    exit /b 1
)

echo.
echo Configuring, building, and running tests...
docker run --rm ^
  -v "%CPP_DIR%:/workspace/brew" ^
  brew-dev ^
  bash -c "cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBREW_ENABLE_PLOTTING=ON -DBREW_BUILD_TESTS=ON && cmake --build /build && ctest --test-dir /build --output-on-failure"

if %ERRORLEVEL% neq 0 (
    echo Build or tests failed.
    exit /b 1
)

echo.
echo Done. Plot outputs are in: %CPP_DIR%\output

endlocal

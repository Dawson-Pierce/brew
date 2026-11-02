#!/usr/bin/env bash
# Validate the role-based public API and native regression suite.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-desktop}" = hardware ]; then
    cmake -S . -B build-hw -G Ninja -DBREW_TARGET=hardware
    cmake --build build-hw --parallel 2
    exit
fi

# All public non-plotting headers must coexist without package aliases.
find include/brew -name '*.hpp' ! -path '*plot*' -print | sort |
    sed 's|^include/|#include "|; s|$|"|' > /tmp/brew_headers.cpp
printf 'int main() {}\n' >> /tmp/brew_headers.cpp
g++ -std=c++20 -fsyntax-only -I include -I /usr/include/eigen3 /tmp/brew_headers.cpp

cmake -S . -B build-all -G Ninja -DBREW_TARGET=desktop \
    -DBREW_ENABLE_PLOTTING=OFF -DBREW_BUILD_EXAMPLES=OFF -DBREW_BUILD_TESTS=ON
cmake --build build-all --parallel 2
ctest --test-dir build-all --output-on-failure

#!/usr/bin/env bash
# Verify the brew C++ library with no MATLAB:
#   1. Each model package compiles standalone (hardware target, BREW_MODELS=<m>):
#      configure + build its libs + syntax-check its umbrella header (works for
#      header-only packages like iggiw too).
#   2. The full library + GoogleTest suite builds and passes.
#   3. The hardware cross-compile-friendly target builds (exceptions off).
#
# Usage (inside the container): build-test.sh [model ...]   (no args -> all)
set -euo pipefail
cd "$(dirname "$0")/.."   # -> brew library root

EIGEN_INC=/usr/include/eigen3

ALL_MODELS=(gaussian ggiw ggiw_orientation iggiw template_pose
            trajectory_gaussian trajectory_ggiw trajectory_ggiw_orientation
            trajectory_iggiw trajectory_template_pose)

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

echo "==== [1/3] Per-package standalone builds (hardware target) ===="
for M in "${MODELS[@]}"; do
    echo "---- package: $M ----"
    cmake -S . -B "build-$M" -G Ninja -DBREW_TARGET=hardware -DBREW_MODELS="$M" >/dev/null
    cmake --build "build-$M"            # builds the package's libs + subsystem deps
    # Verify the package's headers compile standalone (covers header-only packages).
    printf '#include "brew/%s/%s.hpp"\nint main(){ return 0; }\n' "$M" "$M" > "/tmp/chk_$M.cpp"
    g++ -std=c++20 -fsyntax-only -D_USE_MATH_DEFINES -I include -I "$EIGEN_INC" "/tmp/chk_$M.cpp"
    echo "   $M: libs built + umbrella header compiles"
done

echo "==== [2/3] Full desktop build + GoogleTest suite ===="
cmake -S . -B build-all -G Ninja \
    -DBREW_TARGET=desktop -DBREW_ENABLE_PLOTTING=OFF \
    -DBREW_BUILD_EXAMPLES=OFF -DBREW_BUILD_TESTS=ON >/dev/null
cmake --build build-all
ctest --test-dir build-all --output-on-failure

echo "==== [3/3] Hardware target (exceptions off), gaussian only ===="
cmake -S . -B build-hw -G Ninja -DBREW_TARGET=hardware -DBREW_MODELS=gaussian >/dev/null
cmake --build build-hw

echo ""
echo "ALL CHECKS PASSED for: ${MODELS[*]}"

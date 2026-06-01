# brew C++ verification (Docker)

Compiles the **C++ library + GoogleTest suite** with no MATLAB, to confirm each
model package builds and passes standalone (the "is each thing loadable" check)
and that the `hardware` cross-compile target works.

## Run

From the brew library root (this repo):

```bash
docker build -f docker/Dockerfile -t brew-verify .
docker run --rm brew-verify                 # all packages + full suite + hardware
docker run --rm brew-verify gaussian ggiw   # only these packages (step 1)
```

The container performs three checks (see `docker/build-test.sh`):

1. **Per-package standalone build** — for each package: `cmake -DBREW_TARGET=hardware
   -DBREW_MODELS=<m>`, build its libraries, and syntax-check its umbrella header
   `brew/<m>/<m>.hpp` (so header-only packages like `iggiw` are covered too).
2. **Full desktop build + tests** — all packages, plotting off, then `ctest`.
3. **Hardware target** — `BREW_TARGET=hardware` (exceptions off) for `gaussian`.

GoogleTest / EigenRand are fetched via CMake `FetchContent` on first run, so the
build needs network. Eigen comes from the image's `libeigen3-dev`.

## Selecting objects to compile

`BREW_MODELS` is a semicolon list of the packages to build (default: all). For a
hardware/embedded target you typically compile only what you need, e.g.:

```bash
cmake -S . -B build -DBREW_TARGET=hardware -DBREW_MODELS="gaussian;trajectory_gaussian"
```

The MATLAB MEX build (in the parent `brew-lab` wrapper repo) always compiles every
model regardless of `BREW_MODELS`, so the MATLAB side stays comprehensive.

# BREW native verification

From the BREW repository root:

```sh
docker build -f docker/Dockerfile -t brew-verify .
docker run --rm brew-verify
```

The container syntax-checks all public non-plotting headers together, then builds
and runs the desktop C++ tests. GoogleTest and Eigen are installed in the image;
EigenRand is fetched by CMake for the test suite. The application builds do not
fetch these test dependencies.

The library is organized by role and builds as one static library. Density-specific
package targets and `BREW_MODELS` are removed; include the model, merge policy,
filter, and RFS headers needed by your application.

`docker run --rm brew-verify hardware` checks the existing hardware build profile
separately. That profile disables exceptions; it requires filters compatible with
that restriction. The desktop simulations and Python bindings use exceptions.

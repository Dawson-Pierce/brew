# Build/verify the brew C++ library and its tests in isolation, with NO MATLAB.
#
# Compiles the pure C++ library + GoogleTest suite so each model package can be
# confirmed to build and pass standalone (the "is each thing loadable" check),
# and exercises the hardware cross-compile-friendly target.
#
# Build context is the brew library root (this repo):
#   docker build -f docker/Dockerfile -t brew-verify .
#   docker run --rm brew-verify              # all packages + full suite + hardware
#   docker run --rm brew-verify gaussian     # only these packages (step 1)
#
# GoogleTest / EigenRand are fetched by the CMake via FetchContent, so the
# container needs network on first run.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/brew
COPY . /work/brew
RUN chmod +x /work/brew/docker/build-test.sh

ENTRYPOINT ["/work/brew/docker/build-test.sh"]

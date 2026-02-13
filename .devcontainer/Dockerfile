FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG BREW_ENABLE_PLOTTING=ON
ARG BREW_BUILD_TESTS=OFF
ENV BREW_ENABLE_PLOTTING=${BREW_ENABLE_PLOTTING}
ENV BREW_BUILD_TESTS=${BREW_BUILD_TESTS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    ca-certificates \
    git \
    libeigen3-dev \
    libgtest-dev \
    libgmock-dev \
    && if [ "$BREW_ENABLE_PLOTTING" = "ON" ]; then apt-get install -y --no-install-recommends gnuplot-nox; fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/brew

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG BREW_ENABLE_PLOTTING=ON
ARG BREW_BUILD_TESTS=ON

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    libeigen3-dev \
    libgtest-dev \
    libgmock-dev \
    && if [ "$BREW_ENABLE_PLOTTING" = "ON" ]; then apt-get install -y gnuplot-nox; fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# WORKDIR /cpp/build

# RUN cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
#     && ninja

# Use:
# RUN rm -rf CMakeCache.txt CMakeFiles && \
#     cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && \
#     ninja

RUN cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBREW_ENABLE_PLOTTING=${BREW_ENABLE_PLOTTING} \
    -DBREW_BUILD_TESTS=${BREW_BUILD_TESTS} \
    && cmake --build /build

RUN if [ "$BREW_BUILD_TESTS" = "ON" ]; then \
        mkdir -p /workspace/brew/output; \
    fi

CMD ["bash", "-c", "ctest --test-dir /build --output-on-failure && cp -rT /workspace/brew/output /output && echo 'Outputs copied to /output/'"]

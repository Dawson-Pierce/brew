FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    libeigen3-dev \
    libgtest-dev \
    libgmock-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace/cpp

WORKDIR /workspace/cpp/build

# RUN cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
#     && ninja

# Use:
RUN rm -rf CMakeCache.txt CMakeFiles && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    ninja

## Best Practice
# Add `cpp/build/` to your `.dockerignore` file so the build directory isn't copied into the Docker container at all:

CMD ["ctest", "--output-on-failure"]

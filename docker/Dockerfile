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
    && cmake --build /build

CMD ["ctest", "--output-on-failure"]

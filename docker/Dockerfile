FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake ninja-build git ca-certificates libeigen3-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work/brew
COPY . /work/brew
RUN chmod +x /work/brew/docker/build-test.sh
ENTRYPOINT ["/work/brew/docker/build-test.sh"]

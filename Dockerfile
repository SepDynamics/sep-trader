# Use a specific CUDA version for reproducibility
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install all dependencies in a single layer to optimize image size
RUN apt-get update && apt-get install -y \
    lcov sudo gcovr build-essential gcc-14 g++-14 libstdc++-14-dev \
    cmake ninja-build pkg-config curl git python3 python3-pip postgresql-client \
    libpq-dev libhiredis-dev libhwloc-dev libbenchmark-dev \
    nlohmann-json3-dev libglm-dev libglfw3-dev libgl1-mesa-dev libfmt-dev libcurl4-openssl-dev libyaml-cpp-dev \
    autoconf automake libtool \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/jtv/libpqxx.git /tmp/libpqxx && \
    cd /tmp/libpqxx && \
    git checkout 7.9.0 && \
    ./configure --prefix=/usr --disable-documentation && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/libpqxx

# Prefer toolchain v14 by default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 200 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 200 \
    && update-alternatives --set gcc /usr/bin/gcc-14 \
    && update-alternatives --set g++ /usr/bin/g++-14

# Set environment variables for the build
ENV CXX=g++-14
ENV CC=gcc-14
ENV CUDAHOSTCXX=g++-14
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ARG USERNAME=root
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN mkdir -p /etc/sudoers.d && \
    (getent group $USER_GID >/dev/null || groupadd --gid $USER_GID $USERNAME) && \
    (getent passwd $USER_UID >/dev/null || useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash) && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the non-root user
USER $USERNAME

# Set the working directory
WORKDIR /sep

# This Dockerfile creates a build environment.
# The source code is mounted as a volume by the build.sh script, not copied into the image.

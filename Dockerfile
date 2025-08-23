# Use a specific CUDA version for reproducibility
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install all dependencies in a single layer to optimize image size
RUN apt-get update && apt-get install -y \
    lcov sudo gcovr build-essential gcc-11 g++-11 \
    ninja-build pkg-config curl git python3 python3-pip postgresql-client \
    libpq-dev libpqxx-dev libhiredis-dev libhwloc-dev libbenchmark-dev \
    nlohmann-json3-dev libglm-dev libglfw3-dev libgl1-mesa-dev libfmt-dev libcurl4-openssl-dev libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && update-alternatives --set gcc /usr/bin/gcc-11 \
    && update-alternatives --set g++ /usr/bin/g++-11


# Set environment variables for the build
ENV CXX=g++-11
ENV CC=gcc-11
ENV CUDAHOSTCXX=g++-11
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Set up a non-root user to avoid permission issues
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN mkdir -p /etc/sudoers.d && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the non-root user
USER $USERNAME

# Set the working directory
WORKDIR /sep

# This Dockerfile creates a build environment.
# The source code is mounted as a volume by the build.sh script, not copied into the image.

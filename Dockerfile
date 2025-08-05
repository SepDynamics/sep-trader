# Development and runtime environment  
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y \
    lcov \
    gcovr \
    cmake \
    build-essential \
    clang-15 \
    libc++-15-dev \
    libc++abi-15-dev \
    ninja-build \
    pkg-config \
    curl \
    git \
    python3 \
    python3-pip \
    postgresql-client \
    libpq-dev \
    libpqxx-dev \
    libhiredis-dev \
    libhwloc-dev \
    libbenchmark-dev \
    nlohmann-json3-dev \
    libglm-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash sepdsl

# Set working directory
WORKDIR /sep

# Set environment variables for development
ENV CXX=clang++-15
ENV CC=clang-15
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Set default user
USER sepdsl

# Default command
CMD ["/bin/bash"]

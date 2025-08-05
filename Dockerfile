# Use a specific CUDA version for reproducibility
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS builder

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install all dependencies in a single layer to optimize image size
RUN apt-get update && apt-get install -y \
    lcov gcovr cmake build-essential clang-15 libc++-15-dev libc++abi-15-dev \
    ninja-build pkg-config curl git python3 python3-pip postgresql-client \
    libpq-dev libpqxx-dev libhiredis-dev libhwloc-dev libbenchmark-dev \
    nlohmann-json3-dev libglm-dev libglfw3-dev libgl1-mesa-dev libfmt-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /sep

# Set environment variables for the build
ENV CXX=clang++-15
ENV CC=clang-15
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Copy the entire project
COPY . .

# Set up the working directory
WORKDIR /sep

# Install dependencies and build
RUN mkdir build && cd build && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSEP_USE_CUDA=ON && \
    ninja

# Final stage
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04
WORKDIR /sep
COPY --from=builder /sep/build/sep_engine .
CMD ["./sep_engine"]

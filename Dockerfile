# Use a specific CUDA version for reproducibility
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS builder

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install all dependencies in a single layer to optimize image size
RUN apt-get update && apt-get install -y \
    lcov gcovr cmake build-essential gcc-11 g++-11 \
    ninja-build pkg-config curl git python3 python3-pip postgresql-client \
    libpq-dev libpqxx-dev libhiredis-dev libhwloc-dev libbenchmark-dev \
    nlohmann-json3-dev libglm-dev libglfw3-dev libgl1-mesa-dev libfmt-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && update-alternatives --set gcc /usr/bin/gcc-11 \
    && update-alternatives --set g++ /usr/bin/g++-11

# Set up the working directory
WORKDIR /sep

# Set environment variables for the build
ENV CXX=g++-11
ENV CC=gcc-11
ENV CUDAHOSTCXX=g++-11
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Copy the entire project
COPY . .

# Apply comprehensive fixes to resolve std::array header conflicts  
# COMPREHENSIVE FIX: GCC 11 functional header bug - uses unqualified 'array'
# The specific issue is line 1101 in functional header: tuple<std::array<_Tp, _Len>, _Pred> _M_bad_char;
RUN cp /usr/include/c++/11/functional /tmp/functional_backup && \
    sed -i '1s/^/#include <array>\n/' /usr/include/c++/11/functional && \
    sed -i 's/tuple<array<_Tp, _Len>/tuple<std::array<_Tp, _Len>/g' /usr/include/c++/11/functional && \
    sed -i 's/{ array<_Tp, _Len>/{ std::array<_Tp, _Len>/g' /usr/include/c++/11/functional && \
    echo "FIXED FUNCTIONAL HEADER - Added array include and fixed unqualified array usage"

# Fix ALL nlohmann JSON headers with same protection  
RUN find /usr/include/nlohmann -name "*.hpp" -exec sh -c 'printf "#ifdef array\n#undef array\n#endif\n#include <array>\n%s\n" "$(cat "$1")" > "/tmp/$(basename "$1")" && mv "/tmp/$(basename "$1")" "$1"' _ {} \;

# Fix git ownership issues for Docker builds
RUN git config --global --add safe.directory '*'

# Set up the working directory
WORKDIR /sep



# Build environment stage (for development)
FROM builder AS sep_build_env
WORKDIR /sep

# Production build stage  
FROM builder AS production_builder
RUN mkdir build && cd build && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSEP_USE_CUDA=ON \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-11 && \
    ninja

# Final runtime stage
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04
WORKDIR /sep
COPY --from=production_builder /sep/build/sep_engine .
CMD ["./sep_engine"]

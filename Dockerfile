# Use a specific CUDA version for reproducibility
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS builder

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install all dependencies in a single layer to optimize image size
RUN apt-get update && apt-get install -y \
    lcov gcovr cmake build-essential gcc-10 g++-10 \
    ninja-build pkg-config curl git python3 python3-pip postgresql-client \
    libpq-dev libpqxx-dev libhiredis-dev libhwloc-dev libbenchmark-dev \
    nlohmann-json3-dev libglm-dev libglfw3-dev libgl1-mesa-dev libfmt-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

# Set up the working directory
WORKDIR /sep

# Set environment variables for the build
ENV CXX=g++-10
ENV CC=gcc-10
ENV CUDAHOSTCXX=g++-10
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Copy the entire project
COPY . .

# Set up the working directory
WORKDIR /sep

# Fix system headers for build environment
RUN sed -i '1i#include <array>' /usr/include/c++/10/functional && \
    sed -i '4i#include <array>' /usr/include/nlohmann/json.hpp && \
    sed -i 's/_GLIBCXX_STD_C::array/std::array/g' /usr/include/c++/10/functional && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/value_t.hpp && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/conversions/from_json.hpp && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/conversions/to_json.hpp && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/input/input_adapters.hpp && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/input/lexer.hpp && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/input/binary_reader.hpp && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/output/binary_writer.hpp && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/conversions/to_chars.hpp && \
    sed -i '1i #include <array>' /usr/include/nlohmann/detail/output/serializer.hpp

# Build environment stage (for development)
FROM builder AS sep_build_env
WORKDIR /sep

# Production build stage  
FROM builder AS production_builder
RUN mkdir build && cd build && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSEP_USE_CUDA=ON \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-10 && \
    ninja

# Final runtime stage
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04
WORKDIR /sep
COPY --from=production_builder /sep/build/sep_engine .
CMD ["./sep_engine"]

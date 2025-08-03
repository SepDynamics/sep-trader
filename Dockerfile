# Multi-stage Docker build for SEP DSL
# Provides both development environment and production runtime

# Build stage with CUDA support
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    clang-15 \
    libc++-15-dev \
    libc++abi-15-dev \
    pkg-config \
    curl \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up work directory
WORKDIR /sep

# Copy source code
COPY . .

# Build SEP DSL
RUN ./build.sh

# Runtime stage
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash sepdsl

# Copy built artifacts from builder stage
COPY --from=builder /sep/build/src/dsl/sep_dsl_interpreter /usr/local/bin/
COPY --from=builder /sep/build/src/c_api/libsep.so* /usr/local/lib/
COPY --from=builder /sep/src/c_api/sep_c_api.h /usr/local/include/sep/
COPY --from=builder /sep/examples/ /home/sepdsl/examples/
COPY --from=builder /sep/docs/ /home/sepdsl/docs/
COPY --from=builder /sep/README.md /home/sepdsl/

# Set up library path
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/sep.conf && ldconfig

# Create pkg-config file
RUN mkdir -p /usr/local/lib/pkgconfig && \
    echo "prefix=/usr/local" > /usr/local/lib/pkgconfig/sep.pc && \
    echo "exec_prefix=\${prefix}" >> /usr/local/lib/pkgconfig/sep.pc && \
    echo "libdir=\${exec_prefix}/lib" >> /usr/local/lib/pkgconfig/sep.pc && \
    echo "includedir=\${prefix}/include" >> /usr/local/lib/pkgconfig/sep.pc && \
    echo "" >> /usr/local/lib/pkgconfig/sep.pc && \
    echo "Name: SEP DSL" >> /usr/local/lib/pkgconfig/sep.pc && \
    echo "Description: AGI Coherence Framework DSL" >> /usr/local/lib/pkgconfig/sep.pc && \
    echo "Version: 1.0.0" >> /usr/local/lib/pkgconfig/sep.pc && \
    echo "Libs: -L\${libdir} -lsep" >> /usr/local/lib/pkgconfig/sep.pc && \
    echo "Cflags: -I\${includedir}" >> /usr/local/lib/pkgconfig/sep.pc

# Set permissions
RUN chown -R sepdsl:sepdsl /home/sepdsl/

# Switch to non-root user
USER sepdsl
WORKDIR /home/sepdsl

# Test installation
RUN echo 'pattern test { print("SEP DSL Docker build successful!") }' | sep_dsl_interpreter

# Development stage (optional)
FROM builder as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    clang-tools-15 \
    cppcheck \
    doxygen \
    graphviz \
    ruby \
    ruby-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Ruby development gems
RUN gem install bundler rake rake-compiler rspec

# Create development user
RUN useradd -m -s /bin/bash developer && \
    usermod -aG sudo developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up development workspace
WORKDIR /workspace
COPY . .
RUN chown -R developer:developer /workspace

USER developer

# Set up development environment
ENV PATH="/workspace/build/src/dsl:$PATH"
ENV LD_LIBRARY_PATH="/workspace/build/src/c_api:$LD_LIBRARY_PATH"

# Default command
CMD ["/bin/bash"]

# Labels for the image
LABEL maintainer="SEP DSL Team"
LABEL description="SEP DSL - AGI Coherence Framework"
LABEL version="1.0.0"
LABEL url="https://github.com/SepDynamics/sep-dsl"

# Expose ports for future web interfaces
EXPOSE 8080 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD echo 'pattern health { print("OK") }' | sep_dsl_interpreter || exit 1

# Default runtime image
FROM runtime

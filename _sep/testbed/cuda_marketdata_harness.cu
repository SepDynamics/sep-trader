#include "cuda_marketdata_harness.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void doubleKernel(const double* in, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0;
    }
}

std::vector<double> cpuDoubleMid(const std::vector<sep::connectors::MarketData>& data) {
    std::vector<double> out;
    out.reserve(data.size());
    for (const auto& md : data) {
        out.push_back(md.mid * 2.0);
    }
    return out;
}

std::vector<double> gpuDoubleMid(const std::vector<sep::connectors::MarketData>& data) {
    int n = static_cast<int>(data.size());
    std::vector<double> in(n);
    for (int i = 0; i < n; ++i) {
        in[i] = data[i].mid;
    }

    double *d_in = nullptr, *d_out = nullptr;
    size_t bytes = n * sizeof(double);
    cudaError_t err;
    err = cudaMalloc(&d_in, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc d_in failed");
    }
    err = cudaMalloc(&d_out, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_out failed");
    }

    err = cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        throw std::runtime_error("cudaMemcpy to device failed");
    }

    int block = 256;
    int grid = (n + block - 1) / block;
    doubleKernel<<<grid, block>>>(d_in, d_out, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        throw std::runtime_error("Kernel launch failed");
    }

    std::vector<double> out(n);
    err = cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy to host failed");
    }

    return out;
}


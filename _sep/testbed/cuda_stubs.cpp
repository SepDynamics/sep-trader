#ifdef SEP_CPU_ONLY
#include "cuda_marketdata_harness.hpp"
#include "src/engine/internal/cuda_types.hpp"

std::vector<double> gpuDoubleMid(const std::vector<sep::connectors::MarketData>& data) {
    return cpuDoubleMid(data);
}

namespace sep { namespace testbed {

cudaError_t analyzePatterns(const float*, float*, int) {
    return cudaErrorInvalidValue;
}

cudaError_t trainQuantumPatterns(const float*, float*, int, int) {
    return cudaErrorInvalidValue;
}

cudaError_t processMultiPair(const float*, float*, int, int) {
    return cudaErrorInvalidValue;
}

}} // namespace sep::testbed
#endif // SEP_CPU_ONLY

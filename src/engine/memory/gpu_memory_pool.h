#ifndef SEP_GPU_MEMORY_POOL_H
#define SEP_GPU_MEMORY_POOL_H

#include <cstdlib>
#include <cstring>

// This guard checks if the official CUDA runtime header has been included.
// If it HAS NOT, it will define your simple mock types for CPU-only compilation.
// If it HAS, this entire block will be skipped, avoiding the conflict.
#ifndef __CUDA_RUNTIME_H__

// Mock CUDA types for CPU-only compilation
typedef void* cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3

// Mock CUDA API functions
inline cudaError_t cudaMalloc(void** ptr, std::size_t size) { *ptr = std::malloc(size); return *ptr ? cudaSuccess : cudaErrorInvalidValue; }
inline cudaError_t cudaFree(void* ptr) { std::free(ptr); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, int kind) { (void)kind; std::memcpy(dst, src, count); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count, int kind, cudaStream_t stream) { (void)kind; (void)stream; std::memcpy(dst, src, count); return cudaSuccess; }

#endif // End of the guard for mock types
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstddef>

namespace sep {
namespace engine {

struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    bool in_use = false;
    cudaStream_t stream = nullptr;
    
    MemoryBlock() = default;
    MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
};

struct MemoryStats {
    size_t total_allocated = 0;
    size_t total_free = 0;
    size_t peak_usage = 0;
    size_t current_usage = 0;
    size_t num_allocations = 0;
    size_t num_deallocations = 0;
    size_t fragmentation_ratio = 0; // percentage
};

class GPUMemoryPool {
public:
    explicit GPUMemoryPool(size_t initial_pool_size = 256 * 1024 * 1024); // 256MB default
    ~GPUMemoryPool();

    // Core allocation/deallocation
    void* allocate(size_t size, size_t alignment = 256);
    void deallocate(void* ptr);
    
    // Stream-aware operations
    void* allocate_async(size_t size, cudaStream_t stream, size_t alignment = 256);
    void deallocate_async(void* ptr, cudaStream_t stream);
    
    // Memory management
    void defragment();
    void clear();
    void resize_pool(size_t new_size);
    
    // Statistics and monitoring
    MemoryStats get_stats() const;
    void reset_stats();
    
    // Configuration
    void set_auto_defragment(bool enabled, float threshold = 0.5f);
    void set_growth_policy(bool auto_grow, size_t growth_factor = 2);
    
    // Utilities
    bool is_valid_pointer(void* ptr) const;
    size_t get_block_size(void* ptr) const;
    
private:
    mutable std::mutex mutex_;
    std::vector<MemoryBlock> blocks_;
    std::unordered_map<void*, size_t> ptr_to_block_;
    
    void* pool_start_ = nullptr;
    size_t pool_size_ = 0;
    size_t pool_capacity_ = 0;
    
    // Configuration
    bool auto_defragment_ = true;
    float defragment_threshold_ = 0.5f;
    bool auto_grow_ = true;
    size_t growth_factor_ = 2;
    
    // Statistics
    mutable MemoryStats stats_;
    
    // Internal methods
    bool expand_pool(size_t additional_size);
    size_t find_free_block(size_t size, size_t alignment);
    void split_block(size_t block_idx, size_t requested_size);
    void merge_adjacent_blocks();
    void update_stats();
    size_t align_size(size_t size, size_t alignment) const;
};

// RAII wrapper for GPU memory pool allocations
template<typename T>
class PooledGPUMemory {
public:
    explicit PooledGPUMemory(GPUMemoryPool& pool, size_t count = 1);
    ~PooledGPUMemory();
    
    PooledGPUMemory(const PooledGPUMemory&) = delete;
    PooledGPUMemory& operator=(const PooledGPUMemory&) = delete;
    
    PooledGPUMemory(PooledGPUMemory&& other) noexcept;
    PooledGPUMemory& operator=(PooledGPUMemory&& other) noexcept;
    
    T* get() const { return static_cast<T*>(ptr_); }
    size_t count() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }
    
    // Data transfer helpers
    cudaError_t copy_from_host(const T* host_data, size_t elements = 0);
    cudaError_t copy_to_host(T* host_data, size_t elements = 0) const;
    cudaError_t copy_from_host_async(const T* host_data, cudaStream_t stream, size_t elements = 0);
    cudaError_t copy_to_host_async(T* host_data, cudaStream_t stream, size_t elements = 0) const;
    
private:
    GPUMemoryPool* pool_;
    void* ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace engine
} // namespace sep

#endif // SEP_GPU_MEMORY_POOL_H

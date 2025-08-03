#include "gpu_memory_pool.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace sep {
namespace engine {

GPUMemoryPool::GPUMemoryPool(size_t initial_pool_size) 
    : pool_capacity_(initial_pool_size) {
    
    cudaError_t err = cudaMalloc(&pool_start_, pool_capacity_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory pool: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Initialize with one large free block
    blocks_.emplace_back(pool_start_, pool_capacity_);
    pool_size_ = pool_capacity_;
    
    stats_.total_allocated = pool_capacity_;
    stats_.total_free = pool_capacity_;
}

GPUMemoryPool::~GPUMemoryPool() {
    if (pool_start_) {
        cudaFree(pool_start_);
    }
}

void* GPUMemoryPool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t aligned_size = align_size(size, alignment);
    size_t block_idx = find_free_block(aligned_size, alignment);
    
    if (block_idx == SIZE_MAX) {
        // Try defragmentation first
        if (auto_defragment_) {
            merge_adjacent_blocks();
            block_idx = find_free_block(aligned_size, alignment);
        }
        
        // If still no space and auto-grow is enabled, expand pool
        if (block_idx == SIZE_MAX && auto_grow_) {
            size_t needed_size = std::max(aligned_size, pool_capacity_ / growth_factor_);
            if (expand_pool(needed_size)) {
                block_idx = find_free_block(aligned_size, alignment);
            }
        }
        
        if (block_idx == SIZE_MAX) {
            throw std::bad_alloc();
        }
    }
    
    // Split block if necessary
    if (blocks_[block_idx].size > aligned_size) {
        split_block(block_idx, aligned_size);
    }
    
    blocks_[block_idx].in_use = true;
    ptr_to_block_[blocks_[block_idx].ptr] = block_idx;
    
    stats_.num_allocations++;
    stats_.current_usage += aligned_size;
    stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
    
    return blocks_[block_idx].ptr;
}

void GPUMemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = ptr_to_block_.find(ptr);
    if (it == ptr_to_block_.end()) {
        throw std::invalid_argument("Pointer not allocated by this pool");
    }
    
    size_t block_idx = it->second;
    blocks_[block_idx].in_use = false;
    ptr_to_block_.erase(it);
    
    stats_.num_deallocations++;
    stats_.current_usage -= blocks_[block_idx].size;
    
    // Auto-defragment if fragmentation is high
    if (auto_defragment_) {
        update_stats();
        if (stats_.fragmentation_ratio > static_cast<size_t>(defragment_threshold_ * 100)) {
            merge_adjacent_blocks();
        }
    }
}

void* GPUMemoryPool::allocate_async(size_t size, cudaStream_t stream, size_t alignment) {
    void* ptr = allocate(size, alignment);
    
    // Associate with stream for tracking
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_block_.find(ptr);
    if (it != ptr_to_block_.end()) {
        blocks_[it->second].stream = stream;
    }
    
    return ptr;
}

void GPUMemoryPool::deallocate_async(void* ptr, cudaStream_t stream) {
    // For async deallocation, we could defer until stream completion
    // For now, just do immediate deallocation
    deallocate(ptr);
}

void GPUMemoryPool::defragment() {
    std::lock_guard<std::mutex> lock(mutex_);
    merge_adjacent_blocks();
}

void GPUMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Mark all blocks as free
    for (auto& block : blocks_) {
        block.in_use = false;
    }
    ptr_to_block_.clear();
    
    // Merge into one large block
    merge_adjacent_blocks();
    
    stats_.current_usage = 0;
    stats_.num_deallocations += stats_.num_allocations;
}

MemoryStats GPUMemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryStats current_stats = stats_;
    
    // Calculate current fragmentation
    size_t free_blocks = 0;
    for (const auto& block : blocks_) {
        if (!block.in_use) {
            free_blocks++;
        }
    }
    
    if (free_blocks > 1) {
        current_stats.fragmentation_ratio = static_cast<size_t>(
            (static_cast<float>(free_blocks - 1) / blocks_.size()) * 100);
    } else {
        current_stats.fragmentation_ratio = 0;
    }
    
    current_stats.total_free = pool_size_ - stats_.current_usage;
    
    return current_stats;
}

void GPUMemoryPool::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = MemoryStats{};
    stats_.total_allocated = pool_size_;
    stats_.total_free = pool_size_ - stats_.current_usage;
}

void GPUMemoryPool::set_auto_defragment(bool enabled, float threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto_defragment_ = enabled;
    defragment_threshold_ = threshold;
}

void GPUMemoryPool::set_growth_policy(bool auto_grow, size_t growth_factor) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto_grow_ = auto_grow;
    growth_factor_ = growth_factor;
}

bool GPUMemoryPool::is_valid_pointer(void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ptr_to_block_.find(ptr) != ptr_to_block_.end();
}

size_t GPUMemoryPool::get_block_size(void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_block_.find(ptr);
    return (it != ptr_to_block_.end()) ? blocks_[it->second].size : 0;
}

bool GPUMemoryPool::expand_pool(size_t additional_size) {
    void* new_pool = nullptr;
    size_t new_capacity = pool_capacity_ + additional_size;
    
    cudaError_t err = cudaMalloc(&new_pool, new_capacity);
    if (err != cudaSuccess) {
        return false;
    }
    
    // Copy existing data
    err = cudaMemcpy(new_pool, pool_start_, pool_size_, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaFree(new_pool);
        return false;
    }
    
    // Update pointers in blocks
    ptrdiff_t offset = static_cast<char*>(new_pool) - static_cast<char*>(pool_start_);
    for (auto& block : blocks_) {
        block.ptr = static_cast<char*>(block.ptr) + offset;
    }
    
    // Update ptr_to_block map
    std::unordered_map<void*, size_t> new_ptr_map;
    for (const auto& pair : ptr_to_block_) {
        void* new_ptr = static_cast<char*>(pair.first) + offset;
        new_ptr_map[new_ptr] = pair.second;
    }
    ptr_to_block_ = std::move(new_ptr_map);
    
    // Free old pool and update
    cudaFree(pool_start_);
    pool_start_ = new_pool;
    pool_capacity_ = new_capacity;
    
    // Add new free block for the additional space
    void* new_space = static_cast<char*>(pool_start_) + pool_size_;
    blocks_.emplace_back(new_space, additional_size);
    pool_size_ = new_capacity;
    
    stats_.total_allocated = new_capacity;
    
    return true;
}

size_t GPUMemoryPool::find_free_block(size_t size, size_t alignment) {
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (!blocks_[i].in_use && blocks_[i].size >= size) {
            // Check alignment
            uintptr_t addr = reinterpret_cast<uintptr_t>(blocks_[i].ptr);
            if (addr % alignment == 0) {
                return i;
            }
        }
    }
    return SIZE_MAX;
}

void GPUMemoryPool::split_block(size_t block_idx, size_t requested_size) {
    MemoryBlock& block = blocks_[block_idx];
    size_t remaining_size = block.size - requested_size;
    
    if (remaining_size > 0) {
        void* new_ptr = static_cast<char*>(block.ptr) + requested_size;
        blocks_.emplace_back(new_ptr, remaining_size);
        block.size = requested_size;
    }
}

void GPUMemoryPool::merge_adjacent_blocks() {
    std::sort(blocks_.begin(), blocks_.end(), 
              [](const MemoryBlock& a, const MemoryBlock& b) {
                  return a.ptr < b.ptr;
              });
    
    for (size_t i = 0; i < blocks_.size() - 1; ) {
        if (!blocks_[i].in_use && !blocks_[i + 1].in_use) {
            char* end_of_current = static_cast<char*>(blocks_[i].ptr) + blocks_[i].size;
            if (end_of_current == blocks_[i + 1].ptr) {
                // Merge blocks
                blocks_[i].size += blocks_[i + 1].size;
                blocks_.erase(blocks_.begin() + i + 1);
                continue;
            }
        }
        ++i;
    }
}

void GPUMemoryPool::update_stats() {
    // Called with mutex already held
    size_t free_blocks = 0;
    for (const auto& block : blocks_) {
        if (!block.in_use) {
            free_blocks++;
        }
    }
    
    if (free_blocks > 1) {
        stats_.fragmentation_ratio = static_cast<size_t>(
            (static_cast<float>(free_blocks - 1) / blocks_.size()) * 100);
    } else {
        stats_.fragmentation_ratio = 0;
    }
}

size_t GPUMemoryPool::align_size(size_t size, size_t alignment) const {
    return ((size + alignment - 1) / alignment) * alignment;
}

// Template implementations
template<typename T>
PooledGPUMemory<T>::PooledGPUMemory(GPUMemoryPool& pool, size_t count) 
    : pool_(&pool), count_(count) {
    if (count > 0) {
        ptr_ = pool_->allocate(count * sizeof(T), alignof(T));
    }
}

template<typename T>
PooledGPUMemory<T>::~PooledGPUMemory() {
    if (ptr_ && pool_) {
        pool_->deallocate(ptr_);
    }
}

template<typename T>
PooledGPUMemory<T>::PooledGPUMemory(PooledGPUMemory&& other) noexcept 
    : pool_(other.pool_), ptr_(other.ptr_), count_(other.count_) {
    other.pool_ = nullptr;
    other.ptr_ = nullptr;
    other.count_ = 0;
}

template<typename T>
PooledGPUMemory<T>& PooledGPUMemory<T>::operator=(PooledGPUMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_ && pool_) {
            pool_->deallocate(ptr_);
        }
        
        pool_ = other.pool_;
        ptr_ = other.ptr_;
        count_ = other.count_;
        
        other.pool_ = nullptr;
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    return *this;
}

template<typename T>
cudaError_t PooledGPUMemory<T>::copy_from_host(const T* host_data, size_t elements) {
    if (!ptr_ || !host_data) return cudaErrorInvalidValue;
    
    size_t copy_count = (elements == 0) ? count_ : std::min(elements, count_);
    return cudaMemcpy(ptr_, host_data, copy_count * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t PooledGPUMemory<T>::copy_to_host(T* host_data, size_t elements) const {
    if (!ptr_ || !host_data) return cudaErrorInvalidValue;
    
    size_t copy_count = (elements == 0) ? count_ : std::min(elements, count_);
    return cudaMemcpy(host_data, ptr_, copy_count * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
cudaError_t PooledGPUMemory<T>::copy_from_host_async(const T* host_data, cudaStream_t stream, size_t elements) {
    if (!ptr_ || !host_data) return cudaErrorInvalidValue;
    
    size_t copy_count = (elements == 0) ? count_ : std::min(elements, count_);
    return cudaMemcpyAsync(ptr_, host_data, copy_count * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
cudaError_t PooledGPUMemory<T>::copy_to_host_async(T* host_data, cudaStream_t stream, size_t elements) const {
    if (!ptr_ || !host_data) return cudaErrorInvalidValue;
    
    size_t copy_count = (elements == 0) ? count_ : std::min(elements, count_);
    return cudaMemcpyAsync(host_data, ptr_, copy_count * sizeof(T), cudaMemcpyDeviceToHost, stream);
}

// Explicit template instantiations for common types
template class PooledGPUMemory<float>;
template class PooledGPUMemory<double>;
template class PooledGPUMemory<int>;
template class PooledGPUMemory<char>;

} // namespace engine
} // namespace sep

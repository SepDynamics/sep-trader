#pragma once

// Project headers
#include "common/cuda_includes.cuh"
#include "error/cuda_error.cuh"

// Standard library includes
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace sep::cuda::memory {


enum class MemoryType {
    Device,
    Pinned,
    Unified
};

class MemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        MemoryBlock* next;
        MemoryBlock* prev;
        
        MemoryBlock(void* p, size_t s) 
            : ptr(p), size(s), in_use(false), next(nullptr), prev(nullptr) {}
    };

    struct PoolData {
        MemoryBlock* free_list;
        size_t total_memory;
        size_t used_memory;
        
        PoolData() : free_list(nullptr), total_memory(0), used_memory(0) {}
    };

public:
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    void* allocate(size_t size, MemoryType type, cudaStream_t stream = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try to find a suitable block in free list
        MemoryBlock* block = findFreeBlock(size, type);
        if (block) {
            block->in_use = true;
            pools_[type].used_memory += block->size;
            return block->ptr;
        }

        // Allocate new block if none found
        void* ptr = nullptr;
        switch (type) {
            case MemoryType::Device:
                CUDA_CHECK(cudaMalloc(&ptr, size));
                break;
            case MemoryType::Pinned:
                CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
                break;
            case MemoryType::Unified:
                CUDA_CHECK(cudaMallocManaged(&ptr, size));
                break;
        }

        // Create new block and add to pool
        block = new MemoryBlock(ptr, size);
        block->in_use = true;
        insertFreeBlock(block, type);
        
        pools_[type].total_memory += size;
        pools_[type].used_memory += size;
        
        return ptr;
    }

    void deallocate(void* ptr, MemoryType type, cudaStream_t stream = 0) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find the block
        MemoryBlock* block = findBlock(ptr, type);
        if (!block) {
            throw std::runtime_error("Attempt to deallocate unmanaged memory");
        }

        block->in_use = false;
        pools_[type].used_memory -= block->size;
    }

    void preallocate(size_t size, MemoryType type, size_t count = 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (size_t i = 0; i < count; ++i) {
            void* ptr = nullptr;
            switch (type) {
                case MemoryType::Device:
                    CUDA_CHECK(cudaMalloc(&ptr, size));
                    break;
                case MemoryType::Pinned:
                    CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
                    break;
                case MemoryType::Unified:
                    CUDA_CHECK(cudaMallocManaged(&ptr, size));
                    break;
            }
            
            MemoryBlock* block = new MemoryBlock(ptr, size);
            insertFreeBlock(block, type);
            pools_[type].total_memory += size;
        }
    }

    void trim(MemoryType type = MemoryType::Device) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        MemoryBlock* block = pools_[type].free_list;
        while (block) {
            MemoryBlock* next = block->next;
            
            if (!block->in_use) {
                switch (type) {
                    case MemoryType::Device:
                        CUDA_CHECK(cudaFree(block->ptr));
                        break;
                    case MemoryType::Pinned:
                        CUDA_CHECK(cudaFreeHost(block->ptr));
                        break;
                    case MemoryType::Unified:
                        CUDA_CHECK(cudaFree(block->ptr));
                        break;
                }
                
                removeBlock(block, type);
                pools_[type].total_memory -= block->size;
                delete block;
            }
            
            block = next;
        }
    }

    void release() {
        trim(MemoryType::Device);
        trim(MemoryType::Pinned);
        trim(MemoryType::Unified);
    }

    // Statistics
    size_t getUsedMemory(MemoryType type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pools_.at(type).used_memory;
    }

    size_t getTotalMemory(MemoryType type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pools_.at(type).total_memory;
    }

    float getFragmentationRatio(MemoryType type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto& pool = pools_.at(type);
        if (pool.total_memory == 0) return 0.0f;
        return 1.0f - (static_cast<float>(pool.used_memory) / pool.total_memory);
    }

private:
    MemoryPool() = default;
    ~MemoryPool() { release(); }

    std::unordered_map<MemoryType, PoolData> pools_;
    mutable std::mutex mutex_;

    MemoryBlock* findFreeBlock(size_t size, MemoryType type) {
        MemoryBlock* block = pools_[type].free_list;
        while (block) {
            if (!block->in_use && block->size >= size) {
                return block;
            }
            block = block->next;
        }
        return nullptr;
    }

    MemoryBlock* findBlock(void* ptr, MemoryType type) {
        MemoryBlock* block = pools_[type].free_list;
        while (block) {
            if (block->ptr == ptr) {
                return block;
            }
            block = block->next;
        }
        return nullptr;
    }

    void insertFreeBlock(MemoryBlock* block, MemoryType type) {
        if (!pools_[type].free_list) {
            pools_[type].free_list = block;
            return;
        }

        block->next = pools_[type].free_list;
        pools_[type].free_list->prev = block;
        pools_[type].free_list = block;
    }

    void removeBlock(MemoryBlock* block, MemoryType type) {
        if (block->prev) {
            block->prev->next = block->next;
        } else {
            pools_[type].free_list = block->next;
        }

        if (block->next) {
            block->next->prev = block->prev;
        }
    }

    // Prevent copying
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
};

// Type-safe wrapper for pooled memory
template<typename T>
class PooledMemory {
public:
    PooledMemory(size_t count, MemoryType type, cudaStream_t stream = 0)
        : data_(nullptr), size_(count), type_(type), stream_(stream) {
        data_ = static_cast<T*>(MemoryPool::instance().allocate(count * sizeof(T), type, stream));
    }

    ~PooledMemory() {
        if (data_) {
            MemoryPool::instance().deallocate(data_, type_, stream_);
        }
    }

    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return size_; }

private:
    T* data_;
    size_t size_;
    MemoryType type_;
    cudaStream_t stream_;

    // Prevent copying
    PooledMemory(const PooledMemory&) = delete;
    PooledMemory& operator=(const PooledMemory&) = delete;
};

} // namespace sep::cuda::memory
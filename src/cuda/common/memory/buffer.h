#ifndef SEP_CUDA_BUFFER_H
#define SEP_CUDA_BUFFER_H

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "../error/cuda_error.h"

namespace sep {
namespace cuda {

// Base buffer interface
template <typename T>
class Buffer {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;

    virtual ~Buffer() = default;

    // Core buffer interface
    virtual pointer data() = 0;
    virtual const_pointer data() const = 0;
    virtual size_type size() const = 0;
    virtual size_type capacity() const = 0;
    virtual bool empty() const { return size() == 0; }
    
    // Memory operations
    virtual void resize(size_type new_size) = 0;
    virtual void clear() = 0;
    
    // Host-device transfer
    virtual void copyToHost(pointer dst, size_type count, size_type offset = 0) const = 0;
    virtual void copyFromHost(const_pointer src, size_type count, size_type offset = 0) = 0;
    
    // Device-device transfer
    virtual void copyToDevice(pointer dst, size_type count, size_type offset = 0) const = 0;
    virtual void copyFromDevice(const_pointer src, size_type count, size_type offset = 0) = 0;
};

} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_BUFFER_H
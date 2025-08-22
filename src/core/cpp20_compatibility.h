#pragma once

#include <thread>
#include <atomic>
#include <functional>
#include <chrono>

// C++17 compatibility layer for C++20 features
// Simple implementations that work with C++17

namespace sep_compat {

// Simple span implementation for C++17
template<typename T>
class span {
private:
    T* ptr_;
    size_t size_;
    
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    
    constexpr span() noexcept : ptr_(nullptr), size_(0) {}
    
    constexpr span(T* ptr, size_type count) noexcept : ptr_(ptr), size_(count) {}
    
    template<size_t N>
    constexpr span(T (&arr)[N]) noexcept : ptr_(arr), size_(N) {}
    
    template<typename Container>
    constexpr span(Container& cont) noexcept : ptr_(cont.data()), size_(cont.size()) {}
    
    template<typename Container>
    constexpr span(const Container& cont) noexcept : ptr_(cont.data()), size_(cont.size()) {}
    
    constexpr iterator begin() const noexcept { return ptr_; }
    constexpr iterator end() const noexcept { return ptr_ + size_; }
    constexpr const_iterator cbegin() const noexcept { return ptr_; }
    constexpr const_iterator cend() const noexcept { return ptr_ + size_; }
    
    constexpr reference operator[](size_type idx) const { return ptr_[idx]; }
    constexpr reference front() const { return ptr_[0]; }
    constexpr reference back() const { return ptr_[size_ - 1]; }
    constexpr pointer data() const noexcept { return ptr_; }
    
    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type size_bytes() const noexcept { return size_ * sizeof(T); }
    constexpr bool empty() const noexcept { return size_ == 0; }
    
    constexpr span<T> subspan(size_type offset, size_type count = SIZE_MAX) const {
        size_type actual_count = (count == SIZE_MAX) ? size_ - offset : count;
        return span<T>(ptr_ + offset, actual_count);
    }
    
    constexpr span<T> first(size_type count) const {
        return span<T>(ptr_, count);
    }
    
    constexpr span<T> last(size_type count) const {
        return span<T>(ptr_ + size_ - count, count);
    }
};

// Simple stop token implementation for C++17
class stop_token {
private:
    std::atomic<bool>* stop_requested_;
    
public:
    stop_token() : stop_requested_(nullptr) {}
    explicit stop_token(std::atomic<bool>* stop_req) : stop_requested_(stop_req) {}
    
    bool stop_requested() const noexcept {
        return stop_requested_ && stop_requested_->load();
    }
};

class stop_source {
private:
    std::atomic<bool> stop_requested_;
    
public:
    stop_source() : stop_requested_(false) {}
    
    // Delete copy to avoid atomic copy issues
    stop_source(const stop_source&) = delete;
    stop_source& operator=(const stop_source&) = delete;
    
    // Allow move but reset the moved-from object
    stop_source(stop_source&& other) noexcept : stop_requested_(false) {
        // Can't actually move atomic, so just reset both
    }
    
    stop_source& operator=(stop_source&& other) noexcept {
        if (this != &other) {
            stop_requested_.store(false);
        }
        return *this;
    }
    
    stop_token get_token() noexcept {
        return stop_token(&stop_requested_);
    }
    
    bool stop_requested() const noexcept {
        return stop_requested_.load();
    }
    
    void request_stop() noexcept {
        stop_requested_.store(true);
    }
};

// Simple jthread implementation for C++17
class jthread {
private:
    std::thread thread_;
    stop_source stop_source_;
    
public:
    template<typename F, typename... Args>
    explicit jthread(F&& f, Args&&... args) {
        // Start thread without stop token for simplicity
        thread_ = std::thread(std::forward<F>(f), std::forward<Args>(args)...);
    }
    
    jthread() = default;
    
    // Delete copy constructor and assignment
    jthread(const jthread&) = delete;
    jthread& operator=(const jthread&) = delete;
    
    // Move constructor and assignment
    jthread(jthread&& other) noexcept
        : thread_(std::move(other.thread_)), stop_source_(std::move(other.stop_source_)) {}
    
    jthread& operator=(jthread&& other) noexcept {
        if (this != &other) {
            if (joinable()) {
                request_stop();
                join();
            }
            thread_ = std::move(other.thread_);
            stop_source_ = std::move(other.stop_source_);
        }
        return *this;
    }
    
    ~jthread() {
        if (joinable()) {
            request_stop();
            join();
        }
    }
    
    bool joinable() const noexcept {
        return thread_.joinable();
    }
    
    void join() {
        thread_.join();
    }
    
    void detach() {
        thread_.detach();
    }
    
    std::thread::id get_id() const noexcept {
        return thread_.get_id();
    }
    
    void request_stop() noexcept {
        stop_source_.request_stop();
    }
    
    bool stop_requested() const noexcept {
        return stop_source_.stop_requested();
    }
    
    stop_token get_stop_token() noexcept {
        return stop_source_.get_token();
    }
};

} // namespace sep_compat

// Don't pollute std namespace - let code explicitly use sep_compat when needed
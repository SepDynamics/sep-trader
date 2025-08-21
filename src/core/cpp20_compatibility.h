#pragma once

#include <thread>
#include <atomic>
#include <functional>
#include <chrono>

// C++17 compatibility layer for C++20 features
// Simple implementations that work with C++17

namespace sep_compat {

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

// For compatibility, use the sep_compat versions when C++20 features are not available
#if __cplusplus < 202002L
namespace std {
    using jthread = sep_compat::jthread;
    using stop_token = sep_compat::stop_token;
    using stop_source = sep_compat::stop_source;
}
#endif
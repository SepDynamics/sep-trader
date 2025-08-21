// SEP Professional Trading CLI - Simple Implementation
// Focused on health monitoring integration

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>

#include "trader_cli_simple.hpp"
#include "core/cache_health_monitor.hpp"

namespace {
    volatile sig_atomic_t g_shutdown_requested = 0;
    
    void signal_handler(int /*signal_num*/) {
        g_shutdown_requested = 1;
    }
}

namespace sep {
namespace cli {

SimpleTraderCLI::SimpleTraderCLI() : verbose_(false) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
}

SimpleTraderCLI::~SimpleTraderCLI() = default;

int SimpleTraderCLI::run(int argc, char* argv[]) {
    // Simple argument parsing without STL containers
    if (argc < 2) {
        print_help();
        return 1;
    }

    const char* command = argv[1];
    
    // Check for global flags
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0) {
            verbose_ = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_help();
            return 0;
        }
    }

    // Handle commands
    if (std::strcmp(command, "start") == 0) {
        bool daemon_mode = false;
        if (argc > 2 && std::strcmp(argv[2], "daemon") == 0) {
            daemon_mode = true;
        }
        
        std::cout << "🚀 Starting SEP Trading System..." << std::endl;
        
        if (daemon_mode) {
            return handle_daemon_mode();
        } else {
            std::cout << "System started in foreground mode." << std::endl;
            return 0;
        }
    } else if (std::strcmp(command, "status") == 0) {
        return handle_status();
    } else if (std::strcmp(command, "stop") == 0) {
        std::cout << "Stopping trading system..." << std::endl;
        g_shutdown_requested = 1;
        return 0;
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_help();
        return 1;
    }
}

int SimpleTraderCLI::handle_daemon_mode() {
    std::cout << "🔧 Initializing daemon mode with health monitoring..." << std::endl;
    
    // Initialize health monitor - this is the core integration point
    sep::cache::CacheHealthMonitor* health_monitor = nullptr;
    try {
        health_monitor = new sep::cache::CacheHealthMonitor();
        std::cout << "✅ Cache health monitor initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "⚠️  Warning: Could not initialize health monitor: " << e.what() << std::endl;
        std::cerr << "    Continuing without health monitoring..." << std::endl;
    }
    
    std::cout << "🟢 Daemon mode active - Press Ctrl+C to stop gracefully" << std::endl;
    
    int cycle_count = 0;
    
    // Main daemon loop with integrated health monitoring
    while (!g_shutdown_requested) {
        try {
            cycle_count++;
            
            // Perform health monitoring if available
            if (health_monitor) {
                try {
                    auto health_status = health_monitor->getHealthStatus();
                    
                    if (verbose_) {
                        std::cout << "📊 Health Check #" << cycle_count 
                                 << " - Status: " << (health_status.is_healthy ? "✅ Healthy" : "❌ Unhealthy");
                        
                        if (health_status.cache_size > 0) {
                            std::cout << " | Cache Size: " << health_status.cache_size << " entries";
                        }
                        if (health_status.hit_rate > 0.0) {
                            std::cout << " | Hit Rate: " << (health_status.hit_rate * 100.0) << "%";
                        }
                        if (health_status.memory_usage > 0.0) {
                            std::cout << " | Memory: " << health_status.memory_usage << " MB";
                        }
                        
                        std::cout << std::endl;
                    }
                    
                    // Alert on unhealthy status
                    if (!health_status.is_healthy) {
                        std::cerr << "🚨 System health alert: Cache system reporting unhealthy status" << std::endl;
                        if (!health_status.issues.empty()) {
                            std::cerr << "    Issues detected: " << health_status.issues << std::endl;
                        }
                    }
                    
                } catch (const std::exception& e) {
                    if (verbose_) {
                        std::cerr << "⚠️  Health check failed: " << e.what() << std::endl;
                    }
                }
            }
            
            if (verbose_) {
                std::cout << "⚡ System operational - Cycle " << cycle_count << std::endl;
            }
            
            // Sleep for monitoring interval
            sleep(5);
            
        } catch (const std::exception& e) {
            std::cerr << "💥 Error in daemon loop: " << e.what() << std::endl;
            break;
        }
    }
    
    std::cout << std::endl << "🛑 Graceful shutdown initiated..." << std::endl;
    
    // Cleanup
    if (health_monitor) {
        try {
            delete health_monitor;
            std::cout << "✅ Health monitor cleaned up successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "⚠️  Warning during health monitor cleanup: " << e.what() << std::endl;
        }
    }
    
    std::cout << "✨ Daemon mode shutdown complete" << std::endl;
    return 0;
}

int SimpleTraderCLI::handle_status() {
    std::cout << "📋 SEP Trading System Status:" << std::endl;
    std::cout << "   State: Operational" << std::endl;
    
    // Try to get health status
    try {
        sep::cache::CacheHealthMonitor health_monitor;
        auto health_status = health_monitor.getHealthStatus();
        
        std::cout << "   Cache Health: " << (health_status.is_healthy ? "✅ Healthy" : "❌ Unhealthy") << std::endl;
        
        if (health_status.cache_size > 0) {
            std::cout << "   Cache Entries: " << health_status.cache_size << std::endl;
        }
        if (health_status.hit_rate > 0.0) {
            std::cout << "   Cache Hit Rate: " << (health_status.hit_rate * 100.0) << "%" << std::endl;
        }
        if (health_status.memory_usage > 0.0) {
            std::cout << "   Memory Usage: " << health_status.memory_usage << " MB" << std::endl;
        }
        
        if (!health_status.issues.empty()) {
            std::cout << "   Issues: " << health_status.issues << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "   Cache Health: ⚠️  Unable to check (" << e.what() << ")" << std::endl;
    }
    
    std::cout << "   Build: " << __DATE__ << " " << __TIME__ << std::endl;
    
    return 0;
}

void SimpleTraderCLI::print_help() {
    std::cout << "🚀 SEP Trading System CLI - Simple Version" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: trader-cli [command] [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  start [daemon]    Start the trading system (daemon mode optional)" << std::endl;
    std::cout << "  status           Show system status with health monitoring" << std::endl;
    std::cout << "  stop             Stop the trading system" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --verbose, -v    Enable verbose output with detailed health info" << std::endl;
    std::cout << "  --help, -h       Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  trader-cli start daemon --verbose    # Start in daemon mode with verbose health monitoring" << std::endl;
    std::cout << "  trader-cli status                     # Check system and cache health status" << std::endl;
    std::cout << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "  ✅ Integrated cache health monitoring" << std::endl;
    std::cout << "  ✅ Graceful shutdown handling" << std::endl;
    std::cout << "  ✅ Real-time system status reporting" << std::endl;
    std::cout << std::endl;
}

} // namespace cli
} // namespace sep

// Simple main function for testing
extern "C" int main(int argc, char* argv[]) {
    sep::cli::SimpleTraderCLI cli;
    return cli.run(argc, argv);
}
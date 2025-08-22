#pragma once

// Minimal includes - avoiding STL types due to type system pollution
#include <cstdio>

namespace sep {
namespace cli {

class TraderCLI {
public:
    TraderCLI();
    ~TraderCLI();

    int run(int argc, char* argv[]);
    void print_help() const;
    void print_version() const;

private:
    // Method declarations - all using C-style parameters due to type pollution
    int execute_command(const char* command);
    int handle_status() const;
    int handle_pairs() const;
    int handle_config() const;
    int handle_train() const;
    int handle_analyze() const;
    int handle_daemon_mode() const;
    int handle_foreground_mode() const;

    // Simple C-style members
    bool verbose_;
    char config_path_[256];
};

} // namespace cli
} // namespace sep
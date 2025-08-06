#pragma once

#include <string>
#include <vector>
#include "engine/internal/standard_includes.h"

namespace sep {
namespace cli {

struct CLICommand {
    std::string name;
    std::string description;
    std::function<int(const std::vector<std::string>&)> handler;
    std::vector<std::string> subcommands;
};

class TraderCLI {
public:
    TraderCLI();
    ~TraderCLI();

    int run(int argc, char* argv[]);
    void print_help() const;
    void print_version() const;

private:
    void register_commands();
    int execute_command(const std::string& command, const std::vector<std::string>& args);
    
    // Command handlers
    int handle_start(const std::vector<std::string>& args);
    int handle_stop(const std::vector<std::string>& args);
    int handle_status(const std::vector<std::string>& args);
    int handle_pairs(const std::vector<std::string>& args);
    int handle_cache(const std::vector<std::string>& args);
    int handle_config(const std::vector<std::string>& args);
    int handle_logs(const std::vector<std::string>& args);
    int handle_metrics(const std::vector<std::string>& args);

    // Helper methods
    void print_status_table() const;
    void print_pairs_table() const;
    void print_cache_status() const;
    void print_recent_logs(int lines = 20) const;
    void print_performance_metrics() const;
    
    std::map<std::string, CLICommand> commands_;
    bool verbose_;
    std::string config_path_;
};

} // namespace cli
} // namespace sep

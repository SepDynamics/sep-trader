#pragma once

namespace sep {
namespace cli {

class SimpleTraderCLI {
public:
    SimpleTraderCLI();
    ~SimpleTraderCLI();

    int run(int argc, char* argv[]);

private:
    int handle_daemon_mode();
    int handle_status();
    void print_help();
    
    bool verbose_;
};

} // namespace cli
} // namespace sep
#include "trader_cli.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        sep::cli::TraderCLI cli;
        return cli.run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }
}

#include <cstdio>
#include <cstring>
#include "core/dynamic_config_manager.hpp"

namespace sep {
namespace cli {

int run(int argc, char** argv) {
    using sep::config::DynamicConfigManager;
    DynamicConfigManager cfg;

    // Load configuration sources with increasing precedence
    cfg.loadFromFile("config/sep.cfg");
    cfg.loadFromEnvironment();
    cfg.loadFromCommandLine(argc, argv);

    std::string mode = cfg.getStringValue("mode", "sim");

    if (mode == "sim") {
        std::printf("Running simulation mode\n");
    } else if (mode == "live") {
        std::printf("Running live mode\n");
    } else if (mode == "train") {
        std::printf("Running training mode\n");
    } else if (mode == "analyze") {
        std::printf("Running analysis mode\n");
    } else if (mode == "daemon") {
        std::printf("Running daemon mode\n");
    } else {
        std::printf("Unknown mode: %s\n", mode.c_str());
        return 1;
    }

    return 0;
}

} // namespace cli
} // namespace sep

// Expose callable entry for tests
int sep_main(int argc, char** argv) {
    return sep::cli::run(argc, argv);
}

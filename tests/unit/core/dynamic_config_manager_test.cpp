#include <gtest/gtest.h>
#include "core/dynamic_config_manager.hpp"
#include <fstream>
#include <cstdlib>

using namespace sep::config;

TEST(DynamicConfigManagerTest, PrecedenceAndPersistence) {
    DynamicConfigManager mgr;
    // Prepare config file
    const char* file = "test_config.cfg";
    {
        std::ofstream out(file);
        out << "key1=file" << std::endl;
    }
    setenv("SEP_KEY1", "env", 1);
    const char* argv[] = {"prog", "--key1=flag"};
    mgr.loadFromFile(file);
    mgr.loadFromEnvironment();
    mgr.loadFromCommandLine(2, const_cast<char**>(argv));
    EXPECT_EQ(mgr.getStringValue("key1"), "flag");

    mgr.setValue<int>("number", 42);
    ASSERT_TRUE(mgr.saveToFile(file));
    std::ifstream in(file);
    std::string line; std::getline(in, line); // key1 line
    std::getline(in, line); // number line
    EXPECT_NE(line.find("number=42"), std::string::npos);
    unsetenv("SEP_KEY1");
    std::remove(file);
}

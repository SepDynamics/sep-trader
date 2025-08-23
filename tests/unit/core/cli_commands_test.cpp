#include <gtest/gtest.h>

extern "C" {
    void* create_cli_commands();
    void destroy_cli_commands(void* ptr);
    int cli_train_pair(void* ptr, const char* pair);
    int cli_train_all_pairs(void* ptr, int quick_mode);
}

TEST(CLICommandsTest, TrainFunctions) {
    void* cli = create_cli_commands();
    ASSERT_NE(cli, nullptr);
    EXPECT_EQ(cli_train_pair(cli, "EURUSD"), 1);
    EXPECT_EQ(cli_train_all_pairs(cli, 1), 1);
    destroy_cli_commands(cli);
}

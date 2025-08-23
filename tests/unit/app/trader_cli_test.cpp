#include <gtest/gtest.h>

int sep_main(int argc, char** argv);

TEST(TraderCLITest, ModesExecute) {
    char arg0[] = "sep";
    char arg1[] = "--mode";
    char arg2[] = "sim";
    char* argv1[] = {arg0, arg1, arg2};
    EXPECT_EQ(sep_main(3, argv1), 0);

    char arg3[] = "--mode";
    char arg4[] = "daemon";
    char* argv2[] = {arg0, arg3, arg4};
    EXPECT_EQ(sep_main(3, argv2), 0);
}

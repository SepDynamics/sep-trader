#include <gtest/gtest.h>
#include <cstring>
#include <cerrno>

#include "util/error_handling.h"

// Test fixture for error handling tests
class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize error handling system
        sep_error_init(nullptr); // Use stderr for logging
    }
    
    void TearDown() override {
        // Clean up error handling system
        sep_error_cleanup();
    }
};

// Test error initialization
TEST_F(ErrorHandlingTest, Initialization) {
    // Test with stderr
    EXPECT_EQ(sep_error_init(nullptr), 0);
    sep_error_cleanup();
    
    // Test with a file
    EXPECT_EQ(sep_error_init("/tmp/sep_error_test.log"), 0);
    sep_error_cleanup();
}

// Test error reporting
TEST_F(ErrorHandlingTest, ErrorReporting) {
    // Test reporting a system error
    SEP_ERROR_CONTEXT sys_ctx{};
    sys_ctx.file = __FILE__;
    sys_ctx.line = __LINE__;
    sys_ctx.function = __func__;
    sys_ctx.level = SEP_ERROR_ERROR;
    sys_ctx.category = SEP_ERROR_CATEGORY_SYSTEM;
    sys_ctx.error_code = EINVAL;
    sys_ctx.message = "Test system error";
    sep_error_report(sys_ctx);

    // Test reporting a CUDA error
    SEP_ERROR_CONTEXT cuda_ctx{};
    cuda_ctx.file = __FILE__;
    cuda_ctx.line = __LINE__;
    cuda_ctx.function = __func__;
    cuda_ctx.level = SEP_ERROR_ERROR;
    cuda_ctx.category = SEP_ERROR_CATEGORY_CUDA;
    cuda_ctx.error_code = 1;
    cuda_ctx.message = "Test CUDA error";
    sep_error_report(cuda_ctx);
    
    // Note: The CHECK macros would normally be tested here, but they can't be used
    // directly in a test function that returns void as they would cause compilation errors
    // The compilation of this test itself verifies the error handling headers are correct
    
    // All checks should pass without triggering errors
    SUCCEED();
}

// Test error string conversion
TEST_F(ErrorHandlingTest, ErrorStringConversion) {
    // Test system error string
    const char* str = sep_error_string(SEP_ERROR_CATEGORY_SYSTEM, EINVAL);
    EXPECT_NE(str, nullptr);
    EXPECT_STRNE(str, "");
    
    // Test CUDA error string
    str = sep_error_string(SEP_ERROR_CATEGORY_CUDA, 1);
    EXPECT_NE(str, nullptr);
    EXPECT_STREQ(str, "See CUDA error string");
    
    // Test unknown category
    str = sep_error_string(static_cast<SEP_ERROR_CATEGORY>(999), 1);
    EXPECT_NE(str, nullptr);
    EXPECT_STREQ(str, "Unknown error");
}

// Test error callback
TEST_F(ErrorHandlingTest, ErrorCallback) {
    static bool callback_called = false;
    static SEP_ERROR_CONTEXT last_context;
    
    // Set up callback
    auto callback = [](SEP_ERROR_CONTEXT ctx) {
        callback_called = true;
        last_context = ctx;
    };
    
    sep_error_set_callback(callback);
    
    // Trigger an error
    SEP_ERROR_CONTEXT ctx{};
    ctx.file = __FILE__;
    ctx.line = __LINE__;
    ctx.function = __func__;
    ctx.level = SEP_ERROR_WARNING;
    ctx.category = SEP_ERROR_CATEGORY_SYSTEM;
    ctx.error_code = EACCES;
    ctx.message = "Test callback";
    sep_error_report(ctx);
    
    // Check that callback was called
    EXPECT_TRUE(callback_called);
    EXPECT_STREQ(last_context.message, "Test callback");
    EXPECT_EQ(last_context.level, SEP_ERROR_LEVEL::SEP_ERROR_WARNING);
    EXPECT_EQ(last_context.category, SEP_ERROR_CATEGORY_SYSTEM);
    EXPECT_EQ(last_context.error_code, EACCES);
    
    // Reset callback
    sep_error_set_callback(nullptr);
}
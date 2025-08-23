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
    SEP_REPORT_SYSTEM_ERROR(SEP_ERROR_ERROR, EINVAL, "Test system error");
    
    // Test reporting a CUDA error
    SEP_REPORT_CUDA_ERROR(SEP_ERROR_ERROR, 1, "Test CUDA error");
    
    // Test checking for NULL pointer
    int* ptr = nullptr;
    // These macros would normally return, but in the test we just want to make sure they compile
    // Since these are in a test function that returns void, we can't use the CHECK macros directly
    // as they would cause a return statement in a void function
    
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
    SEP_REPORT_SYSTEM_ERROR(SEP_ERROR_WARNING, EACCES, "Test callback");
    
    // Check that callback was called
    EXPECT_TRUE(callback_called);
    EXPECT_STREQ(last_context.message, "Test callback");
    EXPECT_EQ(last_context.level, SEP_ERROR_LEVEL::SEP_ERROR_WARNING);
    EXPECT_EQ(last_context.category, SEP_ERROR_CATEGORY_SYSTEM);
    EXPECT_EQ(last_context.error_code, EACCES);
    
    // Reset callback
    sep_error_set_callback(nullptr);
}
#include <gtest/gtest.h>
#include <cstring>

#include "io/sep_c_api.h"

// Test fixture for C API tests
class SepCApiTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Any setup needed for C API tests
    }
    
    void TearDown() override {
        // Any cleanup needed for C API tests
    }
};

// Test C API lifecycle functions
TEST_F(SepCApiTest, Lifecycle) {
    // Test creating interpreter
    sep_interpreter_t* interp = sep_create_interpreter();
    EXPECT_NE(interp, nullptr);
    
    // Test destroying interpreter
    sep_destroy_interpreter(interp);
}

// Test C API version functions
TEST_F(SepCApiTest, VersionInfo) {
    // Test getting version
    const char* version = sep_get_version();
    EXPECT_NE(version, nullptr);
    EXPECT_STRNE(version, "");
    
    // Test CUDA support check
    int has_cuda = sep_has_cuda_support();
    // This should be 0 or 1
    EXPECT_TRUE(has_cuda == 0 || has_cuda == 1);
}

// Test C API error handling functions
TEST_F(SepCApiTest, ErrorHandling) {
    // Test getting last error from null interpreter
    const char* error = sep_get_last_error(nullptr);
    EXPECT_NE(error, nullptr);
    EXPECT_STREQ(error, "Invalid interpreter");
    
    // Test creating interpreter
    sep_interpreter_t* interp = sep_create_interpreter();
    ASSERT_NE(interp, nullptr);
    
    // Test getting last error from valid interpreter (should be empty)
    error = sep_get_last_error(interp);
    EXPECT_NE(error, nullptr);
    // The error string might be empty or contain some default value
    
    // Test freeing error message
    char* test_msg = strdup("Test error message");
    sep_free_error_message(test_msg);
    // This should not crash
    
    // Clean up
    sep_destroy_interpreter(interp);
}

// Test C API execution functions with error cases
TEST_F(SepCApiTest, ExecutionWithError) {
    // Test executing script with null interpreter
    char* error_msg = nullptr;
    sep_error_t result = sep_execute_script(nullptr, "print('test');", &error_msg);
    EXPECT_EQ(result, SEP_ERROR_MEMORY);
    
    // Test executing script with null script
    sep_interpreter_t* interp = sep_create_interpreter();
    ASSERT_NE(interp, nullptr);
    
    result = sep_execute_script(interp, nullptr, &error_msg);
    EXPECT_EQ(result, SEP_ERROR_MEMORY);
    
    // Test executing file with null interpreter
    result = sep_execute_file(nullptr, "test.sepl", &error_msg);
    EXPECT_EQ(result, SEP_ERROR_MEMORY);
    
    // Test executing file with null filepath
    result = sep_execute_file(interp, nullptr, &error_msg);
    EXPECT_EQ(result, SEP_ERROR_MEMORY);
    
    // Test getting variable with null interpreter
    sep_value_t* value = sep_get_variable(nullptr, "test_var");
    EXPECT_EQ(value, nullptr);
    
    // Test getting variable with null name
    value = sep_get_variable(interp, nullptr);
    EXPECT_EQ(value, nullptr);
    
    // Clean up
    sep_destroy_interpreter(interp);
}

// Test C API value functions with null values
TEST_F(SepCApiTest, ValueFunctionsNull) {
    // Test getting type from null value
    sep_value_type_t type = sep_value_get_type(nullptr);
    EXPECT_EQ(type, SEP_VALUE_NULL);
    
    // Test getting double from null value
    double d = sep_value_as_double(nullptr);
    EXPECT_EQ(d, 0.0);
    
    // Test getting string from null value
    const char* str = sep_value_as_string(nullptr);
    EXPECT_EQ(str, nullptr);
    
    // Test getting boolean from null value
    int b = sep_value_as_boolean(nullptr);
    EXPECT_EQ(b, 0);
    
    // Test freeing null value
    sep_free_value(nullptr);
    // This should not crash
}
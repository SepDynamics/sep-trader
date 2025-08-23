#include <gtest/gtest.h>
#include <string>
#include <cuda_runtime_api.h>

#include "core/result_types.h"

// Test fixture for Result types tests
class ResultTypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Any setup needed for Result types tests
    }
    
    void TearDown() override {
        // Any cleanup needed for Result types tests
    }
};

// Test Result with value
TEST_F(ResultTypesTest, ResultWithValue) {
    // Test with integer value
    sep::Result<int> result = sep::makeSuccess(42);
    EXPECT_TRUE(result.isSuccess());
    EXPECT_FALSE(result.isError());
    EXPECT_EQ(result.value(), 42);
    
    // Test with string value
    sep::Result<std::string> string_result = sep::makeSuccess(std::string("test"));
    EXPECT_TRUE(string_result.isSuccess());
    EXPECT_FALSE(string_result.isError());
    EXPECT_EQ(string_result.value(), "test");
}

// Test Result with error
TEST_F(ResultTypesTest, ResultWithError) {
    // Test with Error object
    sep::Error error(sep::Error::Code::InvalidArgument, "Test error message");
    sep::Result<int> result = sep::makeError<int>(error);
    EXPECT_FALSE(result.isSuccess());
    EXPECT_TRUE(result.isError());
    EXPECT_EQ(result.error().code, sep::Error::Code::InvalidArgument);
    EXPECT_EQ(result.error().message, "Test error message");
    
    // Test with Error code and message
    sep::Result<std::string> string_result = sep::makeError<std::string>(sep::Error::Code::NotFound, "Not found");
    EXPECT_FALSE(string_result.isSuccess());
    EXPECT_TRUE(string_result.isError());
    EXPECT_EQ(string_result.error().code, sep::Error::Code::NotFound);
    EXPECT_EQ(string_result.error().message, "Not found");
}

// Test void Result
TEST_F(ResultTypesTest, VoidResult) {
    // Test successful void result
    sep::Result<void> success_result = sep::makeSuccess();
    EXPECT_TRUE(success_result.isSuccess());
    EXPECT_FALSE(success_result.isError());
    
    // Test void result with error
    sep::Result<void> error_result = sep::makeError(sep::Error::Code::InternalError, "Void error");
    EXPECT_FALSE(error_result.isSuccess());
    EXPECT_TRUE(error_result.isError());
    EXPECT_EQ(error_result.error().code, sep::Error::Code::InternalError);
    EXPECT_EQ(error_result.error().message, "Void error");
}

// Test error string conversion
TEST_F(ResultTypesTest, ErrorStringConversion) {
    // Test success error
    sep::Error success_error(sep::Error::Code::Success);
    EXPECT_EQ(sep::errorToString(success_error), "Success");
    
    // Test invalid argument error
    sep::Error invalid_error(sep::Error::Code::InvalidArgument, "Invalid argument");
    EXPECT_EQ(sep::errorToString(invalid_error), "InvalidArgument: Invalid argument");
    
    // Test error without message
    sep::Error no_message_error(sep::Error::Code::NotFound);
    EXPECT_EQ(sep::errorToString(no_message_error), "NotFound");
}

// Test SEPResult string conversion
TEST_F(ResultTypesTest, SEPResultStringConversion) {
    EXPECT_EQ(sep::resultToString(sep::SEPResult::SUCCESS), "SUCCESS");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::INVALID_ARGUMENT), "INVALID_ARGUMENT");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::NOT_FOUND), "NOT_FOUND");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::PROCESSING_ERROR), "PROCESSING_ERROR");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::NOT_INITIALIZED), "NOT_INITIALIZED");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::CUDA_ERROR), "CUDA_ERROR");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::UNKNOWN_ERROR), "UNKNOWN_ERROR");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::RESOURCE_UNAVAILABLE), "RESOURCE_UNAVAILABLE");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::OPERATION_FAILED), "OPERATION_FAILED");
    EXPECT_EQ(sep::resultToString(sep::SEPResult::NOT_IMPLEMENTED), "NOT_IMPLEMENTED");
    EXPECT_EQ(sep::resultToString(static_cast<sep::SEPResult>(999)), "UNKNOWN");
}

// Test fromSEPResult conversion
TEST_F(ResultTypesTest, FromSEPResultConversion) {
    // Test successful conversion
    sep::Result<int> success_result = sep::fromSEPResult<int>(sep::SEPResult::SUCCESS);
    EXPECT_TRUE(success_result.isSuccess());
    
    // Test error conversion
    sep::Result<int> error_result = sep::fromSEPResult<int>(sep::SEPResult::INVALID_ARGUMENT, "Invalid argument");
    EXPECT_TRUE(error_result.isError());
    EXPECT_EQ(error_result.error().code, sep::Error::Code::InvalidArgument);
    EXPECT_EQ(error_result.error().message, "Invalid argument");
    
    // Test void result conversion
    sep::Result<void> void_success_result = sep::fromSEPResult<void>(sep::SEPResult::SUCCESS);
    EXPECT_TRUE(void_success_result.isSuccess());
    
    sep::Result<void> void_error_result = sep::fromSEPResult<void>(sep::SEPResult::PROCESSING_ERROR, "Processing error");
    EXPECT_TRUE(void_error_result.isError());
    EXPECT_EQ(void_error_result.error().code, sep::Error::Code::ProcessingError);
    EXPECT_EQ(void_error_result.error().message, "Processing error");
}

// Test fromCudaError conversion
TEST_F(ResultTypesTest, FromCudaErrorConversion) {
    // Test successful conversion
    sep::Result<void> success_result = sep::fromCudaError(cudaSuccess);
    EXPECT_TRUE(success_result.isSuccess());
    
    // Test error conversion
    sep::Result<void> error_result = sep::fromCudaError(cudaErrorInvalidValue, "Invalid value");
    EXPECT_TRUE(error_result.isError());
    EXPECT_EQ(error_result.error().code, sep::Error::Code::InvalidArgument);
    EXPECT_EQ(error_result.error().message, "Invalid value: invalid argument");
}
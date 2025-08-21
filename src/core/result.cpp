#include "core/result_types.h"

#include <cuda_runtime_api.h>
#include <string>

namespace sep {

std::string resultToString(SEPResult result) {
    switch (result) {
        case SEPResult::SUCCESS: return "SUCCESS";
        case SEPResult::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case SEPResult::NOT_FOUND: return "NOT_FOUND";
        case SEPResult::ALREADY_EXISTS: return "ALREADY_EXISTS";
        case SEPResult::PROCESSING_ERROR: return "PROCESSING_ERROR";
        case SEPResult::NOT_INITIALIZED: return "NOT_INITIALIZED";
        case SEPResult::CUDA_ERROR: return "CUDA_ERROR";
        case SEPResult::UNKNOWN_ERROR: return "UNKNOWN_ERROR";
        case SEPResult::RESOURCE_UNAVAILABLE: return "RESOURCE_UNAVAILABLE";
        case SEPResult::OPERATION_FAILED: return "OPERATION_FAILED";
        case SEPResult::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
        default: return "UNKNOWN";
    }
}

std::string errorToString(const Error& error) {
    std::string code_str;
    switch (error.code) {
        case Error::Code::Success: code_str = "Success"; break;
        case Error::Code::InvalidArgument: code_str = "InvalidArgument"; break;
        case Error::Code::NotFound: code_str = "NotFound"; break;
        case Error::Code::ProcessingError: code_str = "ProcessingError"; break;
        case Error::Code::InternalError: code_str = "InternalError"; break;
        case Error::Code::NotInitialized: code_str = "NotInitialized"; break;
        case Error::Code::CudaError: code_str = "CudaError"; break;
        case Error::Code::UnknownError: code_str = "UnknownError"; break;
        case Error::Code::ResourceUnavailable: code_str = "ResourceUnavailable"; break;
        case Error::Code::OperationFailed: code_str = "OperationFailed"; break;
        case Error::Code::NotImplemented: code_str = "NotImplemented"; break;
        case Error::Code::AlreadyExists: code_str = "AlreadyExists"; break;
        case Error::Code::Internal: code_str = "Internal"; break;
    }
    if (!error.message.empty()) {
        return code_str + ": " + error.message;
    }
    return code_str;
}

template<typename T>
Result<T> fromSEPResult(SEPResult result, const std::string& message) {
    if (result == SEPResult::SUCCESS) {
        return makeSuccess(T{});
    }
    return makeError<T>(Error(result, message));
}

template<>
Result<void> fromSEPResult<void>(SEPResult result, const std::string& message) {
    if (result == SEPResult::SUCCESS) {
        return makeSuccess();
    }
    return makeError(Error(result, message));
}

static Error::Code mapCudaError(cudaError_t err) {
    switch (err) {
        case cudaSuccess: return Error::Code::Success;
        case cudaErrorInvalidValue: return Error::Code::InvalidArgument;
        case cudaErrorMemoryAllocation: return Error::Code::ResourceUnavailable;
        case cudaErrorInitializationError: return Error::Code::NotInitialized;
        default: return Error::Code::CudaError;
    }
}

Result<void> fromCudaError(cudaError_t err, const std::string& context) {
    if (err == cudaSuccess) {
        return makeSuccess();
    }
    std::string msg = context;
    if (!context.empty()) {
        msg += ": ";
    }
    msg += cudaGetErrorString(err);
    return makeError(Error(mapCudaError(err), msg));
}

// Explicit template instantiations
template Result<void> fromSEPResult<void>(SEPResult, const std::string&);

} // namespace sep


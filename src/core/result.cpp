#include "core/result_types.h"

namespace sep::core {

const char* resultToString(SEPResult result) {
    switch (result) {
        case SEPResult::SUCCESS:
            return "SUCCESS";
        case SEPResult::FAILURE:
            return "FAILURE";
        case SEPResult::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case SEPResult::OUT_OF_MEMORY:
            return "OUT_OF_MEMORY";
        case SEPResult::CUDA_ERROR:
            return "CUDA_ERROR";
        case SEPResult::FILE_NOT_FOUND:
            return "FILE_NOT_FOUND";
        case SEPResult::NETWORK_ERROR:
            return "NETWORK_ERROR";
        case SEPResult::TIMEOUT:
            return "TIMEOUT";
        case SEPResult::NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case SEPResult::ALREADY_EXISTS:
            return "ALREADY_EXISTS";
        case SEPResult::NOT_FOUND:
            return "NOT_FOUND";
        case SEPResult::NOT_IMPLEMENTED:
            return "NOT_IMPLEMENTED";
        case SEPResult::PROCESSING_ERROR:
            return "PROCESSING_ERROR";
        case SEPResult::RUNTIME_ERROR:
            return "RUNTIME_ERROR";
        case SEPResult::UNKNOWN_ERROR:
            return "UNKNOWN_ERROR";
        default:
            return "UNKNOWN";
    }
}

} // namespace sep::core

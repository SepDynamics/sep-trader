#include "sep_c_api.h"
#include "dsl/runtime/runtime.h"
#include "engine/facade/facade.h"
#include <string>
#include <memory>
#include <cstring>
#include <any>

// The C struct is just a wrapper for the C++ class
struct sep_interpreter {
    std::unique_ptr<dsl::runtime::DSLRuntime> cpp_runtime;
    std::string last_error;
    
    sep_interpreter() : cpp_runtime(std::make_unique<dsl::runtime::DSLRuntime>()) {}
};

struct sep_value {
    std::any cpp_value;
    std::string cached_string; // For string values, to ensure lifetime
    
    sep_value(const std::any& val) : cpp_value(val) {}
};

// Implementation of the C functions
sep_interpreter_t* sep_create_interpreter() {
    try {
        // Initialize the engine ONCE
        auto& facade = sep::engine::EngineFacade::getInstance();
        facade.initialize();
        
        return new sep_interpreter();
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void sep_destroy_interpreter(sep_interpreter_t* interp) {
    delete interp;
}

sep_error_t sep_execute_script(sep_interpreter_t* interp, const char* script_source, char** error_msg) {
    if (!interp || !script_source) {
        return SEP_ERROR_MEMORY;
    }
    
    try {
        std::string source(script_source);
        interp->cpp_runtime->execute(source);
        return SEP_SUCCESS;
    } catch (const std::exception& e) {
        interp->last_error = e.what();
        if (error_msg) {
            *error_msg = strdup(interp->last_error.c_str());
        }
        return SEP_ERROR_RUNTIME;
    }
}

sep_error_t sep_execute_file(sep_interpreter_t* interp, const char* filepath, char** error_msg) {
    if (!interp || !filepath) {
        return SEP_ERROR_MEMORY;
    }
    
    try {
        std::string path(filepath);
        interp->cpp_runtime->execute_file(path);
        return SEP_SUCCESS;
    } catch (const std::exception& e) {
        interp->last_error = e.what();
        if (error_msg) {
            *error_msg = strdup(interp->last_error.c_str());
        }
        return SEP_ERROR_RUNTIME;
    }
}

sep_value_t* sep_get_variable(sep_interpreter_t* interp, const char* name) {
    if (!interp || !name) {
        return nullptr;
    }
    
    try {
        std::string var_name(name);
        
        // Check if this is a pattern member access (contains a dot)
        std::size_t dot_pos = var_name.find('.');
        if (dot_pos != std::string::npos) {
            // Split into pattern name and member name
            std::string pattern_name = var_name.substr(0, dot_pos);
            std::string member_name = var_name.substr(dot_pos + 1);
            
            if (!interp->cpp_runtime->has_variable(pattern_name)) {
                return nullptr;
            }
            
            // Get the pattern result
            auto pattern_value = interp->cpp_runtime->get_variable(pattern_name);
            
            // Try to cast to PatternResult (unordered_map<string, Value>)
            typedef std::unordered_map<std::string, std::any> PatternResult;
            try {
                PatternResult pattern_result = std::any_cast<PatternResult>(pattern_value);
                auto it = pattern_result.find(member_name);
                if (it != pattern_result.end()) {
                    return new sep_value(it->second);
                }
                return nullptr;
            } catch (const std::bad_any_cast&) {
                interp->last_error = "Variable '" + pattern_name + "' is not a pattern";
                return nullptr;
            }
        } else {
            // Direct variable access
            if (!interp->cpp_runtime->has_variable(var_name)) {
                return nullptr;
            }
            auto value = interp->cpp_runtime->get_variable(var_name);
            return new sep_value(value);
        }
    } catch (const std::exception& e) {
        interp->last_error = e.what();
        return nullptr;
    }
}

sep_value_type_t sep_value_get_type(sep_value_t* value) {
    if (!value || !value->cpp_value.has_value()) {
        return SEP_VALUE_NULL;
    }
    
    const std::type_info& type = value->cpp_value.type();
    if (type == typeid(double) || type == typeid(float) || type == typeid(int)) {
        return SEP_VALUE_NUMBER;
    } else if (type == typeid(std::string)) {
        return SEP_VALUE_STRING;
    } else if (type == typeid(bool)) {
        return SEP_VALUE_BOOLEAN;
    } else {
        return SEP_VALUE_NULL;
    }
}

double sep_value_as_double(sep_value_t* value) {
    if (!value || !value->cpp_value.has_value()) {
        return 0.0;
    }
    
    try {
        const std::type_info& type = value->cpp_value.type();
        if (type == typeid(double)) {
            return std::any_cast<double>(value->cpp_value);
        } else if (type == typeid(float)) {
            return static_cast<double>(std::any_cast<float>(value->cpp_value));
        } else if (type == typeid(int)) {
            return static_cast<double>(std::any_cast<int>(value->cpp_value));
        }
    } catch (const std::bad_any_cast&) {
        return 0.0;
    }
    return 0.0;
}

const char* sep_value_as_string(sep_value_t* value) {
    if (!value || !value->cpp_value.has_value()) {
        return nullptr;
    }
    
    try {
        if (value->cpp_value.type() == typeid(std::string)) {
            value->cached_string = std::any_cast<std::string>(value->cpp_value);
        } else {
            // Convert to string representation
            const std::type_info& type = value->cpp_value.type();
            if (type == typeid(double)) {
                value->cached_string = std::to_string(std::any_cast<double>(value->cpp_value));
            } else if (type == typeid(int)) {
                value->cached_string = std::to_string(std::any_cast<int>(value->cpp_value));
            } else if (type == typeid(bool)) {
                value->cached_string = std::any_cast<bool>(value->cpp_value) ? "true" : "false";
            } else {
                value->cached_string = "unknown";
            }
        }
    } catch (const std::bad_any_cast&) {
        value->cached_string = "error";
    }
    return value->cached_string.c_str();
}

int sep_value_as_boolean(sep_value_t* value) {
    if (!value || !value->cpp_value.has_value()) {
        return 0;
    }
    
    try {
        const std::type_info& type = value->cpp_value.type();
        if (type == typeid(bool)) {
            return std::any_cast<bool>(value->cpp_value) ? 1 : 0;
        } else if (type == typeid(double)) {
            return std::any_cast<double>(value->cpp_value) != 0.0 ? 1 : 0;
        } else if (type == typeid(int)) {
            return std::any_cast<int>(value->cpp_value) != 0 ? 1 : 0;
        }
    } catch (const std::bad_any_cast&) {
        return 0;
    }
    return 0;
}

void sep_free_value(sep_value_t* value) {
    delete value;
}

void sep_free_error_message(char* error_msg) {
    free(error_msg);
}

const char* sep_get_last_error(sep_interpreter_t* interp) {
    if (!interp) {
        return "Invalid interpreter";
    }
    return interp->last_error.c_str();
}

const char* sep_get_version() {
    return "1.0.0-alpha";
}

int sep_has_cuda_support() {
#ifdef CUDA_FOUND
    return 1;
#else
    return 0;
#endif
}

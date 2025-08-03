#ifndef SEP_C_API_H
#define SEP_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointers to hide C++ implementation details
typedef struct sep_interpreter sep_interpreter_t;
typedef struct sep_value sep_value_t;

// Error codes
typedef enum {
    SEP_SUCCESS = 0,
    SEP_ERROR_PARSE = 1,
    SEP_ERROR_RUNTIME = 2,
    SEP_ERROR_ENGINE = 3,
    SEP_ERROR_MEMORY = 4
} sep_error_t;

// Value types
typedef enum {
    SEP_VALUE_NUMBER = 0,
    SEP_VALUE_STRING = 1,
    SEP_VALUE_BOOLEAN = 2,
    SEP_VALUE_NULL = 3
} sep_value_type_t;

// Lifecycle
sep_interpreter_t* sep_create_interpreter();
void sep_destroy_interpreter(sep_interpreter_t* interp);

// Execution
// Returns SEP_SUCCESS on success, error code on failure. Fills error_msg.
sep_error_t sep_execute_script(sep_interpreter_t* interp, const char* script_source, char** error_msg);
sep_error_t sep_execute_file(sep_interpreter_t* interp, const char* filepath, char** error_msg);

// Getting results
sep_value_t* sep_get_variable(sep_interpreter_t* interp, const char* name);
sep_value_type_t sep_value_get_type(sep_value_t* value);
double sep_value_as_double(sep_value_t* value);
const char* sep_value_as_string(sep_value_t* value);
int sep_value_as_boolean(sep_value_t* value);
void sep_free_value(sep_value_t* value);

// Error handling
void sep_free_error_message(char* error_msg);
const char* sep_get_last_error(sep_interpreter_t* interp);

// Version info
const char* sep_get_version();
int sep_has_cuda_support();

#ifdef __cplusplus
}
#endif

#endif // SEP_C_API_H

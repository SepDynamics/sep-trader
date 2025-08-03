#include <ruby.h>
#include <sep/sep_c_api.h>

static VALUE mSEP;
static VALUE cInterpreter;

// Free function for interpreter objects
static void interpreter_free(void *ptr) {
    if (ptr) {
        sep_destroy_interpreter((sep_interpreter_t*)ptr);
    }
}

// Allocation function for interpreter objects
static VALUE interpreter_alloc(VALUE klass) {
    sep_interpreter_t *interp = sep_create_interpreter();
    if (!interp) {
        rb_raise(rb_eRuntimeError, "Failed to create SEP interpreter");
    }
    return Data_Wrap_Struct(klass, 0, interpreter_free, interp);
}

// SEP::Interpreter.new
static VALUE interpreter_initialize(VALUE self) {
    return self;
}

// interpreter.execute(script_source)
static VALUE interpreter_execute(VALUE self, VALUE script_source) {
    sep_interpreter_t *interp;
    Data_Get_Struct(self, sep_interpreter_t, interp);
    
    Check_Type(script_source, T_STRING);
    const char *source = RSTRING_PTR(script_source);
    
    char *error_msg = NULL;
    sep_error_t result = sep_execute_script(interp, source, &error_msg);
    
    if (result != SEP_SUCCESS) {
        VALUE error = rb_str_new_cstr(error_msg ? error_msg : "Unknown error");
        if (error_msg) {
            sep_free_error_message(error_msg);
        }
        rb_raise(rb_eRuntimeError, "SEP execution failed: %s", RSTRING_PTR(error));
    }
    
    return Qtrue;
}

// interpreter.execute_file(filepath)
static VALUE interpreter_execute_file(VALUE self, VALUE filepath) {
    sep_interpreter_t *interp;
    Data_Get_Struct(self, sep_interpreter_t, interp);
    
    Check_Type(filepath, T_STRING);
    const char *path = RSTRING_PTR(filepath);
    
    char *error_msg = NULL;
    sep_error_t result = sep_execute_file(interp, path, &error_msg);
    
    if (result != SEP_SUCCESS) {
        VALUE error = rb_str_new_cstr(error_msg ? error_msg : "Unknown error");
        if (error_msg) {
            sep_free_error_message(error_msg);
        }
        rb_raise(rb_eRuntimeError, "SEP file execution failed: %s", RSTRING_PTR(error));
    }
    
    return Qtrue;
}

// interpreter.get_variable(name)
static VALUE interpreter_get_variable(VALUE self, VALUE name) {
    sep_interpreter_t *interp;
    Data_Get_Struct(self, sep_interpreter_t, interp);
    
    Check_Type(name, T_STRING);
    const char *var_name = RSTRING_PTR(name);
    
    sep_value_t *value = sep_get_variable(interp, var_name);
    if (!value) {
        return Qnil;
    }
    
    VALUE ruby_value;
    sep_value_type_t type = sep_value_get_type(value);
    
    switch (type) {
        case SEP_VALUE_NUMBER:
            ruby_value = rb_float_new(sep_value_as_double(value));
            break;
        case SEP_VALUE_STRING:
            ruby_value = rb_str_new_cstr(sep_value_as_string(value));
            break;
        case SEP_VALUE_BOOLEAN:
            ruby_value = sep_value_as_boolean(value) ? Qtrue : Qfalse;
            break;
        default:
            ruby_value = Qnil;
            break;
    }
    
    sep_free_value(value);
    return ruby_value;
}

// SEP.version
static VALUE sep_version(VALUE self) {
    return rb_str_new_cstr(sep_get_version());
}

// SEP.has_cuda?
static VALUE sep_has_cuda(VALUE self) {
    return sep_has_cuda_support() ? Qtrue : Qfalse;
}

// This function is called by Ruby when the gem is loaded
void Init_sep_dsl() {
    mSEP = rb_define_module("SEP");
    
    cInterpreter = rb_define_class_under(mSEP, "Interpreter", rb_cObject);
    rb_define_alloc_func(cInterpreter, interpreter_alloc);
    rb_define_method(cInterpreter, "initialize", interpreter_initialize, 0);
    rb_define_method(cInterpreter, "execute", interpreter_execute, 1);
    rb_define_method(cInterpreter, "execute_file", interpreter_execute_file, 1);
    rb_define_method(cInterpreter, "get_variable", interpreter_get_variable, 1);
    rb_define_alias(cInterpreter, "[]", "get_variable");
    
    // Module methods
    rb_define_singleton_method(mSEP, "version", sep_version, 0);
    rb_define_singleton_method(mSEP, "has_cuda?", sep_has_cuda, 0);
}

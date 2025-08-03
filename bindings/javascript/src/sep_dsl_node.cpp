#include <node.h>
#include <node_object_wrap.h>
#include <v8.h>
#include "sep_c_api.h"

namespace sep_dsl {

using v8::Context;
using v8::Function;
using v8::FunctionCallbackInfo;
using v8::FunctionTemplate;
using v8::Isolate;
using v8::Local;
using v8::NewStringType;
using v8::Object;
using v8::ObjectTemplate;
using v8::Persistent;
using v8::String;
using v8::Value;
using v8::Exception;

class DSLInterpreter : public node::ObjectWrap {
public:
    static void Init(Local<Object> exports);

private:
    explicit DSLInterpreter();
    ~DSLInterpreter();

    static void New(const FunctionCallbackInfo<Value>& args);
    static void Execute(const FunctionCallbackInfo<Value>& args);
    static void GetVariable(const FunctionCallbackInfo<Value>& args);
    
    static Persistent<Function> constructor;
    void* interpreter_;  // SEP interpreter handle
};

Persistent<Function> DSLInterpreter::constructor;

DSLInterpreter::DSLInterpreter() {
    interpreter_ = sep_create_interpreter();
    if (!interpreter_) {
        // Constructor will handle the error
    }
}

DSLInterpreter::~DSLInterpreter() {
    if (interpreter_) {
        sep_destroy_interpreter(interpreter_);
    }
}

void DSLInterpreter::Init(Local<Object> exports) {
    Isolate* isolate = exports->GetIsolate();
    Local<Context> context = isolate->GetCurrentContext();

    // Prepare constructor template
    Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New);
    tpl->SetClassName(String::NewFromUtf8(
        isolate, "DSLInterpreter", NewStringType::kNormal).ToLocalChecked());
    tpl->InstanceTemplate()->SetInternalFieldCount(1);

    // Prototype methods
    NODE_SET_PROTOTYPE_METHOD(tpl, "execute", Execute);
    NODE_SET_PROTOTYPE_METHOD(tpl, "getVariable", GetVariable);

    Local<Function> constructor_func = tpl->GetFunction(context).ToLocalChecked();
    constructor.Reset(isolate, constructor_func);
    exports->Set(context, String::NewFromUtf8(
        isolate, "DSLInterpreter", NewStringType::kNormal).ToLocalChecked(),
        constructor_func).FromJust();
}

void DSLInterpreter::New(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();
    Local<Context> context = isolate->GetCurrentContext();

    if (args.IsConstructCall()) {
        // Invoked as constructor: `new DSLInterpreter()`
        DSLInterpreter* obj = new DSLInterpreter();
        
        if (!obj->interpreter_) {
            isolate->ThrowException(Exception::Error(
                String::NewFromUtf8(isolate, "Failed to create DSL interpreter",
                                   NewStringType::kNormal).ToLocalChecked()));
            return;
        }
        
        obj->Wrap(args.This());
        args.GetReturnValue().Set(args.This());
    } else {
        // Invoked as plain function, turn into construct call
        const int argc = 0;
        Local<Value> argv[1] = {};
        Local<Function> cons = Local<Function>::New(isolate, constructor);
        Local<Object> result = cons->NewInstance(context, argc, argv).ToLocalChecked();
        args.GetReturnValue().Set(result);
    }
}

void DSLInterpreter::Execute(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    if (args.Length() < 1 || !args[0]->IsString()) {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Script must be a string",
                               NewStringType::kNormal).ToLocalChecked()));
        return;
    }

    DSLInterpreter* obj = ObjectWrap::Unwrap<DSLInterpreter>(args.Holder());
    
    if (!obj->interpreter_) {
        isolate->ThrowException(Exception::Error(
            String::NewFromUtf8(isolate, "Interpreter not initialized",
                               NewStringType::kNormal).ToLocalChecked()));
        return;
    }

    String::Utf8Value script(isolate, args[0]);
    
    int result = sep_execute_script(obj->interpreter_, *script);
    if (result != 0) {
        isolate->ThrowException(Exception::Error(
            String::NewFromUtf8(isolate, "DSL script execution failed",
                               NewStringType::kNormal).ToLocalChecked()));
        return;
    }

    args.GetReturnValue().SetUndefined();
}

void DSLInterpreter::GetVariable(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    if (args.Length() < 1 || !args[0]->IsString()) {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Variable name must be a string",
                               NewStringType::kNormal).ToLocalChecked()));
        return;
    }

    DSLInterpreter* obj = ObjectWrap::Unwrap<DSLInterpreter>(args.Holder());
    
    if (!obj->interpreter_) {
        isolate->ThrowException(Exception::Error(
            String::NewFromUtf8(isolate, "Interpreter not initialized",
                               NewStringType::kNormal).ToLocalChecked()));
        return;
    }

    String::Utf8Value name(isolate, args[0]);
    
    char value_buffer[1024];
    int result = sep_get_variable(obj->interpreter_, *name, value_buffer, sizeof(value_buffer));
    if (result != 0) {
        isolate->ThrowException(Exception::Error(
            String::NewFromUtf8(isolate, "Variable not found",
                               NewStringType::kNormal).ToLocalChecked()));
        return;
    }

    args.GetReturnValue().Set(String::NewFromUtf8(
        isolate, value_buffer, NewStringType::kNormal).ToLocalChecked());
}

// Module initialization
void InitAll(Local<Object> exports) {
    DSLInterpreter::Init(exports);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, InitAll)

}  // namespace sep_dsl

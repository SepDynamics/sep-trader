#include <Python.h>
#include "sep_c_api.h"

// Python wrapper for the SEP DSL interpreter
typedef struct {
    PyObject_HEAD
    void* interpreter;  // SEPInterpreter handle
} DSLInterpreterObject;

// Forward declarations
static PyTypeObject DSLInterpreterType;
static void DSLInterpreter_dealloc(DSLInterpreterObject *self);
static PyObject *DSLInterpreter_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int DSLInterpreter_init(DSLInterpreterObject *self, PyObject *args, PyObject *kwds);
static PyObject *DSLInterpreter_execute(DSLInterpreterObject *self, PyObject *args);
static PyObject *DSLInterpreter_get_variable(DSLInterpreterObject *self, PyObject *args);

// Method definitions
static PyObject *DSLInterpreter_execute(DSLInterpreterObject *self, PyObject *args) {
    const char *script;
    if (!PyArg_ParseTuple(args, "s", &script)) {
        return NULL;
    }
    
    if (!self->interpreter) {
        PyErr_SetString(PyExc_RuntimeError, "Interpreter not initialized");
        return NULL;
    }
    
    int result = sep_execute_script(self->interpreter, script);
    if (result != 0) {
        PyErr_SetString(PyExc_RuntimeError, "DSL script execution failed");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject *DSLInterpreter_get_variable(DSLInterpreterObject *self, PyObject *args) {
    const char *name;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        return NULL;
    }
    
    if (!self->interpreter) {
        PyErr_SetString(PyExc_RuntimeError, "Interpreter not initialized");
        return NULL;
    }
    
    char value_buffer[1024];
    int result = sep_get_variable(self->interpreter, name, value_buffer, sizeof(value_buffer));
    if (result != 0) {
        PyErr_SetString(PyExc_KeyError, "Variable not found");
        return NULL;
    }
    
    return PyUnicode_FromString(value_buffer);
}

static PyObject *DSLInterpreter_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    DSLInterpreterObject *self;
    self = (DSLInterpreterObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->interpreter = NULL;
    }
    return (PyObject *)self;
}

static int DSLInterpreter_init(DSLInterpreterObject *self, PyObject *args, PyObject *kwds) {
    self->interpreter = sep_create_interpreter();
    if (!self->interpreter) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create DSL interpreter");
        return -1;
    }
    return 0;
}

static void DSLInterpreter_dealloc(DSLInterpreterObject *self) {
    if (self->interpreter) {
        sep_destroy_interpreter(self->interpreter);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// Method table
static PyMethodDef DSLInterpreter_methods[] = {
    {"execute", (PyCFunction)DSLInterpreter_execute, METH_VARARGS,
     "Execute DSL script"},
    {"get_variable", (PyCFunction)DSLInterpreter_get_variable, METH_VARARGS,
     "Get variable value from DSL context"},
    {NULL}  // Sentinel
};

// Type definition
static PyTypeObject DSLInterpreterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sep_dsl._sep_dsl.DSLInterpreter",
    .tp_doc = "SEP DSL Interpreter",
    .tp_basicsize = sizeof(DSLInterpreterObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = DSLInterpreter_new,
    .tp_init = (initproc)DSLInterpreter_init,
    .tp_dealloc = (destructor)DSLInterpreter_dealloc,
    .tp_methods = DSLInterpreter_methods,
};

// Module definition
static PyModuleDef sep_dsl_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_sep_dsl",
    .m_doc = "SEP DSL C Extension",
    .m_size = -1,
};

// Module initialization
PyMODINIT_FUNC PyInit__sep_dsl(void) {
    PyObject *m;
    
    if (PyType_Ready(&DSLInterpreterType) < 0)
        return NULL;
    
    m = PyModule_Create(&sep_dsl_module);
    if (m == NULL)
        return NULL;
    
    Py_INCREF(&DSLInterpreterType);
    if (PyModule_AddObject(m, "DSLInterpreter", (PyObject *)&DSLInterpreterType) < 0) {
        Py_DECREF(&DSLInterpreterType);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}

# SEP DSL Standard Library (stdlib) Architecture

This document outlines the architecture for the SEP DSL's standard library (`stdlib`), designed to be modular, extensible, and maintainable.

## 1. Design Goals

The primary goals of this architecture are:

- **Modularity**: Group related functions into logical modules (e.g., `math`, `statistical`) to improve organization and reduce cognitive load.
- **Extensibility**: Make it straightforward to add new functions or entire new modules without modifying existing, unrelated code.
- **Maintainability**: A clean, organized structure makes the codebase easier to understand, debug, and test.
- **Clear Integration**: Provide a single, clear entry point for registering all standard library functions with the DSL runtime.

## 2. Directory Structure

All standard library code resides within `src/dsl/stdlib/`. The structure is organized by category, as defined in the project roadmap:

```
src/dsl/stdlib/
├── aggregation/
│   ├── aggregation.cpp
│   └── aggregation.h
├── core_primitives.cpp
├── core_primitives.h
├── data_transformation/
│   ├── data_transformation.cpp
│   └── data_transformation.h
├── math/
│   ├── math.cpp
│   └── math.h
├── pattern_matching/
│   ├── pattern_matching.cpp
│   └── pattern_matching.h
├── statistical/
│   ├── statistical.cpp
│   └── statistical.h
├── stdlib.cpp
├── stdlib.h
└── time_series/
    ├── time_series.cpp
    └── time_series.h
```

## 3. Core Components

### Module Files

Each functional category has its own subdirectory containing a header (`.h`) and a source (`.cpp`) file.

- **Header (`module.h`)**: Declares a single registration function for that module (e.g., `void register_math(Runtime& runtime);`).
- **Source (`module.cpp`)**: Implements the registration function. This is where the individual built-in functions for the module will be defined and registered with the `Runtime`.

### Central Integration Files

The root of the `stdlib` directory contains two key files for integrating all the modules.

- **`stdlib.h`**: A central header that includes all the individual module headers.
- **`stdlib.cpp`**: Implements the `register_all(Runtime& runtime)` function. This function serves as the single entry point for the DSL runtime to initialize the entire standard library by calling the registration function from each module.

## 4. How to Add a New Function

To add a new built-in function (e.g., `cos`):

1.  **Identify the Module**: Determine the correct module for the function (e.g., `math`).
2.  **Implement the Function**: Add the C++ implementation of the `cos` function inside `src/dsl/stdlib/math/math.cpp`.
3.  **Register the Function**: Inside the `register_math` function in the same file, add a call to register your new `cos` function with the `runtime` object.

This modular approach ensures that developers can work on different parts of the standard library simultaneously with minimal conflicts and a clear separation of concerns.
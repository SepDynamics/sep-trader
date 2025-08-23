# Utility Helpers

This directory documents core helpers powering the SEP DSL environment.

## Compiler
- `compiler.*` converts parsed DSL into bytecode via `BytecodeProgram` and
  `CompiledProgram` wrappers.
- Simplified AST stubs underpin compilation; stream mocks previously used for
  testing have been removed to keep behaviour deterministic.

## Interpreter
- `interpreter.*` executes AST nodes against a scoped `Environment` and exposes
  built‑in functions via a registration map.
- Module loading and export tracking remain TODO items and some legacy function
  hooks persist until the built‑ins are fully migrated.

## Memory Tiers
- `memory_tier*` and `memory_tier_manager*` provide a tiered allocation system
  (STM, MTM, LTM) with promotion, fragmentation metrics and optional
  serialization.

## Mocked or Deprecated Components
- `array_protection.h` is deprecated in favor of the consolidated
  `sep_precompiled` header.
- `stdlib.cpp` carries placeholder modules and interpreter notes flag incomplete
  pattern input handling.

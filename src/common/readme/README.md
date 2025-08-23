# Common Headers

Shared headers live here to coordinate symbol handling across modules.

## Namespace Protection
- `namespace_protection.hpp` guards standard names like `std`, `string` and
  `cout` from macro pollution.
- Include it before and after any third‑party headers that redefine core
  symbols to push/restore the original macros.

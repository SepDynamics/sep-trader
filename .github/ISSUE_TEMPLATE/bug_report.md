---
name: Bug report
about: Create a report to help us improve SEP DSL
title: '[BUG] '
labels: 'bug'
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## Environment
- **OS**: (e.g., Ubuntu 22.04, Windows 11, macOS 13)
- **SEP DSL Version**: (e.g., 1.0.0)
- **CUDA Version**: (e.g., 12.2, or "None" if not using CUDA)
- **Compiler**: (e.g., g++, g++-11)
- **Installation Method**: (Docker, local build, package manager)

## To Reproduce
Steps to reproduce the behavior:

1. Create file with this content:
```sep
pattern example {
    // Your minimal reproduction case here
}
```

2. Run command: `./sep_dsl_interpreter example.sep`
3. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Error Output
```
Paste any error messages, stack traces, or unexpected output here
```

## Minimal Reproduction Case
Please provide the smallest possible SEP DSL code that reproduces the issue:

```sep
pattern minimal_case {
    // Minimal code that demonstrates the bug
}
```

## Additional Context
- Are you using language bindings? (Ruby, Python, etc.)
- Does this happen with different input data?
- Any recent changes to your environment?
- Screenshots or additional logs if helpful

## Possible Solution
If you have ideas about what might be causing the issue or how to fix it, please share them here.

## Checklist
- [ ] I have searched existing issues for this bug
- [ ] I have provided a minimal reproduction case
- [ ] I have included my environment details
- [ ] I have tested with the latest version

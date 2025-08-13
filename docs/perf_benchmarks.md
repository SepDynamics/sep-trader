# Performance Benchmarks

The refactored pattern analysis kernel processes four elements per thread,
reducing launch overhead and improving memory coalescing.

## Benchmark Setup

The script `/_sep/testbed/quantum/pattern_bench.py` simulates the legacy
scalar kernel and the new vectorized version over 1,000,000 data points.

## Results

```
old_time 1.077639
new_time 0.368933
speedup 2.92x
```

The optimized kernel delivers roughly a **3× throughput improvement** over the
previous implementation.

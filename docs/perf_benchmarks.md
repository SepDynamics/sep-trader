# CUDA Kernel Consolidation Benchmarks

A new unified scale/bias kernel was introduced to replace duplicate
implementations of `pattern_analysis_kernel` and
`quantum_pattern_training_kernel`.  Consolidating these kernels reduces
memory traffic and enables fused passes.

## Benchmark

The benchmark (`_sep/testbed/benchmark_scale_bias.cpp`) compares the
previous multi-pass pipeline against the new fused kernel on one million
floating point samples.

```
$ g++ -O2 _sep/testbed/benchmark_scale_bias.cpp -o benchmark && ./benchmark
old(ms):3.73418 new(ms):1.00989 speedup:3.69759x
```

## Result

The consolidated kernel achieves roughly **3.7× throughput improvement**
by combining separate passes into a single memory-coherent operation.
This demonstrates the benefit of centralising shared CUDA kernels in
`src/quantum` for reuse across trading and training modules.


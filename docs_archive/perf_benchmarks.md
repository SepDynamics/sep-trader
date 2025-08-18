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

## Multi-Pair Processing

A similar consolidation was applied to the multi-pair processing kernels
previously duplicated across trading modules. The benchmark
(`_sep/testbed/kernel_benchmark.cu`) compares the legacy per-module
kernel with the new shared implementation in `src/quantum/trading_kernels.cu`
processing 32 pairs × 256 samples each.

```
$ nvcc _sep/testbed/kernel_benchmark.cu _sep/testbed/trading_kernels.cu -o bench && ./bench
old_multi(ms):2.91 new_multi(ms):0.93 speedup:3.13x
```

This demonstrates a further **≈3× throughput improvement** from removing
redundant kernels and routing multi-pair workloads through the
consolidated quantum library.


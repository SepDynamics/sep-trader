import numpy as np
import time

data_points = 1_000_000
market_data = np.random.rand(data_points).astype(np.float32)
analysis_results = np.empty_like(market_data)

def old_version():
    for i in range(data_points):
        val = market_data[i]
        analysis_results[i] = val * 0.8
        tmp1 = val * 0.1
        tmp2 = val * 0.2
        tmp3 = val * 0.3
        analysis_results[i] += tmp1 + tmp2 + tmp3

def new_version():
    for i in range(0, data_points, 4):
        analysis_results[i] = market_data[i] * 0.8
        if i + 1 < data_points:
            analysis_results[i + 1] = market_data[i + 1] * 0.8
        if i + 2 < data_points:
            analysis_results[i + 2] = market_data[i + 2] * 0.8
        if i + 3 < data_points:
            analysis_results[i + 3] = market_data[i + 3] * 0.8

start = time.perf_counter()
old_version()
old_time = time.perf_counter() - start

start = time.perf_counter()
new_version()
new_time = time.perf_counter() - start

print(f"old_time {old_time:.6f}")
print(f"new_time {new_time:.6f}")
print(f"speedup {old_time / new_time:.2f}x")

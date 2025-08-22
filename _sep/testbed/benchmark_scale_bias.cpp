#include <chrono>
#include <iostream>
#include <vector>

void old_pipeline(const std::vector<float>& in, std::vector<float>& out1, std::vector<float>& out2) {
    for (size_t i = 0; i < in.size(); ++i) {
        out1[i] = in[i] * 0.8f;
    }
    for (size_t i = 0; i < in.size(); ++i) {
        out2[i] = in[i] * 0.5f + 0.5f;
    }
    for (size_t i = 0; i < in.size(); ++i) {
        out1[i] += out2[i];
    }
    for (size_t i = 0; i < in.size(); ++i) {
        out2[i] *= 0.9f;
    }
    for (size_t i = 0; i < in.size(); ++i) {
        out1[i] -= 0.1f;
    }
    for (size_t i = 0; i < in.size(); ++i) {
        out2[i] += 0.2f;
    }
}

void new_pipeline(const std::vector<float>& in, std::vector<float>& out1, std::vector<float>& out2) {
    for (size_t i = 0; i < in.size(); ++i) {
        float x = in[i];
        out1[i] = x * 0.8f;
        out2[i] = x * 0.5f + 0.5f;
    }
}

int main() {
    const size_t N = 1 << 20;
    std::vector<float> input(N, 1.0f), out1(N), out2(N);

    auto start = std::chrono::high_resolution_clock::now();
    old_pipeline(input, out1, out2);
    auto mid = std::chrono::high_resolution_clock::now();
    new_pipeline(input, out1, out2);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> old_ms = mid - start;
    std::chrono::duration<double, std::milli> new_ms = end - mid;
    std::cout << "old(ms):" << old_ms.count() << " new(ms):" << new_ms.count()
              << " speedup:" << old_ms.count()/new_ms.count() << "x" << std::endl;
    return 0;
}


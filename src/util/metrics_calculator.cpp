#include "metrics_calculator.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream> // For debug logging, can be removed later

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sep {
namespace metrics_library {

// --- Helper Structs and Enums (Copied from qfh.h and trajectory.h) ---
enum class QFHState {
    NULL_STATE,
    FLIP,
    RUPTURE
};

struct QFHEvent {
    size_t index;
    QFHState state;
    uint8_t bit_prev;
    uint8_t bit_curr;
};

struct QFHAggregateEvent {
    size_t index;
    QFHState state;
    int count;
};

namespace bitspace {
struct DampedValue {
    double final_value;
    double confidence; // Used as stability
    bool converged;
    std::vector<double> path;
};
} // namespace bitspace

// --- Helper Functions (Copied and adapted from qfh.cpp) ---

std::vector<QFHEvent> transform_rich(const std::vector<uint8_t>& bits) {
    std::vector<QFHEvent> result;
    if (bits.size() < 2) {
        return result;
    }
    result.reserve(bits.size() - 1);
    for (size_t i = 1; i < bits.size(); ++i) {
        uint8_t prev = bits[i - 1];
        uint8_t curr = bits[i];
        if ((prev != 0 && prev != 1) || (curr != 0 && curr != 1)) {
            // Handle invalid bits, or assume valid input for this library
            return {}; 
        }
        if (prev == 0 && curr == 0) {
            result.push_back({i - 1, QFHState::NULL_STATE, prev, curr});
        } else if ((prev == 0 && curr == 1) || (prev == 1 && curr == 0)) {
            result.push_back({i - 1, QFHState::FLIP, prev, curr});
        } else if (prev == 1 && curr == 1) {
            result.push_back({i - 1, QFHState::RUPTURE, prev, curr});
        }
    }
    return result;
}

std::vector<QFHAggregateEvent> aggregate(const std::vector<QFHEvent>& events) {
    if (events.empty()) {
        return {};
    }
    std::vector<QFHAggregateEvent> aggregated;
    aggregated.push_back({events[0].index, events[0].state, 1});
    for (size_t i = 1; i < events.size(); ++i) {
        if (events[i].state == aggregated.back().state) {
            aggregated.back().count++;
        } else {
            aggregated.push_back({events[i].index, events[i].state, 1});
        }
    }
    return aggregated;
}

// Helper function to calculate cosine similarity between two vectors
double calculateCosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0;
    }
    
    return dot_product / (norm_a * norm_b);
}

bitspace::DampedValue integrateFutureTrajectories(const std::vector<uint8_t>& bitstream, size_t current_index) {
    bitspace::DampedValue damped_value;
    
    if (current_index >= bitstream.size()) {
        damped_value.final_value = 0.0;
        damped_value.confidence = 0.0;
        damped_value.converged = false;
        return damped_value;
    }
    
    // Step 1: Calculate dynamic decay factor λ based on local entropy and coherence
    size_t window_size = std::min(static_cast<size_t>(20), bitstream.size() - current_index);
    std::vector<uint8_t> local_window(bitstream.begin() + current_index, 
                                      bitstream.begin() + current_index + window_size);
    
    auto local_events = transform_rich(local_window);
    double local_entropy = 0.5; // Default entropy
    double local_coherence = 0.5; // Default coherence
    
    if (!local_events.empty()) {
        int null_count = 0, flip_count = 0, rupture_count = 0;
        for (const auto& event : local_events) {
            switch (event.state) {
                case QFHState::NULL_STATE: null_count++; break;
                case QFHState::FLIP: flip_count++; break;
                case QFHState::RUPTURE: rupture_count++; break;
            }
        }
        
        float total = static_cast<float>(local_events.size());
        float null_ratio = null_count / total;
        float flip_ratio = flip_count / total;
        float rupture_ratio = rupture_count / total;
        
        auto safe_log2 = [](float x) -> float { return (x > 0.0f) ? std::log2(x) : 0.0f; };
        local_entropy = -(null_ratio * safe_log2(null_ratio) + flip_ratio * safe_log2(flip_ratio) + rupture_ratio * safe_log2(rupture_ratio));
        local_entropy = std::fmax(0.05, std::fmin(1.0, local_entropy / 1.585));
        local_coherence = 1.0 - local_entropy;
    }
    
    const double k1 = 0.30;  // Entropy weight
    const double k2 = 0.20;  // Coherence weight
    double lambda = k1 * local_entropy + k2 * (1.0 - local_coherence);
    lambda = std::fmax(0.01, std::fmin(1.0, lambda));  // Constrain to reasonable range
    
    // Step 2: Integrate future trajectories with exponential damping
    double accumulated_value = 0.0;
    double current_bit_val = static_cast<double>(bitstream[current_index]);
    
    damped_value.path.clear();
    damped_value.path.reserve(bitstream.size() - current_index);
    damped_value.path.push_back(current_bit_val);
    
    for (size_t j = current_index + 1; j < bitstream.size(); ++j) {
        double future_bit_val = static_cast<double>(bitstream[j]);
        double time_difference = static_cast<double>(j - current_index);
        double weight = std::exp(-lambda * time_difference);
        
        double contribution = (future_bit_val - current_bit_val) * weight;
        accumulated_value += contribution;
        
        damped_value.path.push_back(accumulated_value);
    }
    
    damped_value.final_value = accumulated_value;
    
    // Step 3: Calculate preliminary confidence based on trajectory consistency
    if (damped_value.path.size() > 2) {
        double trajectory_variance = 0.0;
        double mean_trajectory = 0.0;
        
        for (double val : damped_value.path) {
            mean_trajectory += val;
        }
        mean_trajectory /= damped_value.path.size();
        
        for (double val : damped_value.path) {
            trajectory_variance += std::pow(val - mean_trajectory, 2);
        }
        trajectory_variance /= damped_value.path.size();
        
        double stability_score = 1.0 / (1.0 + trajectory_variance);
        damped_value.confidence = std::fmax(0.0, std::fmin(1.0, stability_score));
    } else {
        damped_value.confidence = 0.5;  // Default confidence for insufficient data
    }
    
    damped_value.converged = (damped_value.confidence > 0.7);
    
    return damped_value;
}

double matchKnownPaths(const std::vector<double>& current_path) {
    if (current_path.size() < 3) {
        return 0.5;  // Default confidence for insufficient data
    }
    
    double best_similarity = 0.0;
    
    // Pattern 1: Exponential decay similarity
    std::vector<double> exponential_pattern;
    exponential_pattern.reserve(current_path.size());
    double initial_value = current_path[0];
    for (size_t i = 0; i < current_path.size(); ++i) {
        double expected_value = initial_value * std::exp(-0.1 * static_cast<double>(i));
        exponential_pattern.push_back(expected_value);
    }
    
    double exp_similarity = calculateCosineSimilarity(current_path, exponential_pattern);
    best_similarity = std::max(best_similarity, exp_similarity);
    
    // Pattern 2: Linear trend similarity
    if (current_path.size() >= 2) {
        std::vector<double> linear_pattern;
        linear_pattern.reserve(current_path.size());
        double slope = (current_path.back() - current_path.front()) / (current_path.size() - 1);
        for (size_t i = 0; i < current_path.size(); ++i) {
            double expected_value = current_path[0] + slope * static_cast<double>(i);
            linear_pattern.push_back(expected_value);
        }
        
        double linear_similarity = calculateCosineSimilarity(current_path, linear_pattern);
        best_similarity = std::max(best_similarity, linear_similarity);
    }
    
    // Pattern 3: Oscillating pattern similarity (sine wave approximation)
    std::vector<double> oscillating_pattern;
    oscillating_pattern.reserve(current_path.size());
    double amplitude = (current_path.back() - current_path.front()) / 2.0;
    double mean_value = (current_path.back() + current_path.front()) / 2.0;
    for (size_t i = 0; i < current_path.size(); ++i) {
        double expected_value = mean_value + amplitude * std::sin(2.0 * M_PI * i / current_path.size());
        oscillating_pattern.push_back(expected_value);
    }
    
    double osc_similarity = calculateCosineSimilarity(current_path, oscillating_pattern);
    best_similarity = std::max(best_similarity, osc_similarity);
    
    return std::fmax(0.0, std::fmin(1.0, best_similarity));
}

// --- Main Metrics Calculation Function ---

MetricsResult calculateMetrics(const std::vector<uint8_t>& bitstream) {
    MetricsResult result_metrics;
    
    // Mimic QFHBasedProcessor::analyze logic
    
    // Transform bits to events
    auto events = transform_rich(bitstream);
    
    // Count event types
    int null_state_count = 0;
    int flip_count = 0;
    int rupture_count = 0;

    for (const auto& event : events) {
        switch (event.state) {
            case QFHState::NULL_STATE: null_state_count++; break;
            case QFHState::FLIP: flip_count++; break;
            case QFHState::RUPTURE: rupture_count++; break;
        }
    }
    
    // Calculate ratios
    float rupture_ratio = 0.0f;
    float flip_ratio = 0.0f;
    if (!events.empty()) {
        rupture_ratio = static_cast<float>(rupture_count) / static_cast<float>(events.size());
        flip_ratio = static_cast<float>(flip_count) / static_cast<float>(events.size());
    }

    // Calculate entropy
    if (!events.empty()) {
        float null_ratio = static_cast<float>(null_state_count) / static_cast<float>(events.size());
        
        auto safe_log2 = [](float x) -> float { return (x > 0.0f) ? std::log2(x) : 0.0f; };
        
        result_metrics.entropy = -(null_ratio * safe_log2(null_ratio) +
                                  flip_ratio * safe_log2(flip_ratio) +
                                  rupture_ratio * safe_log2(rupture_ratio));
        
        result_metrics.entropy = std::fmax(0.05f, std::fmin(1.0f, result_metrics.entropy / 1.585f));
    } else {
        result_metrics.entropy = 0.0f; // Default for empty events
    }
    
    // Calculate pattern-based coherence
    float pattern_coherence = (1.0f - result_metrics.entropy) * 1.2f;
    float stability_coherence_component = (1.0f - rupture_ratio) * 1.1f; // Renamed to avoid confusion with final stability
    float flip_coherence_component = (1.0f - flip_ratio) * 1.05f;
    
    float raw_coherence = pattern_coherence * 0.5f + stability_coherence_component * 0.3f + flip_coherence_component * 0.2f;
    result_metrics.coherence = std::fmax(0.0f, std::fmin(1.0f, raw_coherence * raw_coherence * 1.1f));

    // Integrate future trajectories for damping and stability
    if (!bitstream.empty() && bitstream.size() > 10) {
        bitspace::DampedValue dv = integrateFutureTrajectories(bitstream, 0);
        
        double trajectory_confidence = matchKnownPaths(dv.path);
        
        float trajectory_coherence = static_cast<float>(trajectory_confidence);
        
        // Blend pattern-based coherence with trajectory-based coherence
        result_metrics.coherence = 0.3f * trajectory_coherence + 0.7f * result_metrics.coherence;
        
        // Apply damped final value to influence coherence stability
        if (std::abs(dv.final_value) < 2.0) {
            float stability_factor = 1.0f / (1.0f + 0.1f * std::abs(static_cast<float>(dv.final_value)));
            result_metrics.coherence = result_metrics.coherence * stability_factor;
        }
        
        // Ensure coherence stays within valid range
        result_metrics.coherence = std::fmax(0.0f, std::fmin(1.0f, result_metrics.coherence));

        // Use the confidence from integrateFutureTrajectories as the stability metric
        result_metrics.stability = static_cast<float>(dv.confidence);

    } else {
        // Default stability if bitstream is too small for trajectory analysis
        result_metrics.stability = 0.5f; 
    }
    
    return result_metrics;
}

} // namespace metrics_library
} // namespace sep

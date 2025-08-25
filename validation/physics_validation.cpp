#include "physics_validation.hpp"
#include <cstring>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <type_traits>

namespace sep::physics_validation {

// Utility functions
static void quickSort(float* arr, int low, int high) {
    if (low < high) {
        float pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                float temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        
        float temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        
        int pi = i + 1;
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

static float clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// SignalToBinaryConverter Implementation
SignalToBinaryConverter::SignalToBinaryConverter(BitMappingType type)
    : mapping_type_(type), buffer_size_(0) {
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        quantile_thresholds_[i] = 0.0f;
    }
    std::memset(signal_buffer_, 0, sizeof(signal_buffer_));
}

BinaryState SignalToBinaryConverter::convert(const float* signal, size_t signal_length, size_t window_size) {
    if (signal == nullptr || signal_length == 0) {
        BinaryState empty_state{};
        return empty_state;
    }
    
    switch (mapping_type_) {
        case BitMappingType::D1_DERIVATIVE_SIGN:
            return applyD1Mapping(signal, signal_length);
        case BitMappingType::D2_QUANTILE_THRESHOLDS:
            return applyD2Mapping(signal, signal_length);
        case BitMappingType::D3_MULTIBAND_FEATURES:
            return applyD3Mapping(signal, signal_length);
        default:
            return applyD1Mapping(signal, signal_length);
    }
}

BinaryState SignalToBinaryConverter::convertPoint(float value) {
    if (buffer_size_ < 1024) {
        signal_buffer_[buffer_size_++] = value;
    } else {
        // Shift buffer left and add new value
        for (size_t i = 0; i < 1023; ++i) {
            signal_buffer_[i] = signal_buffer_[i + 1];
        }
        signal_buffer_[1023] = value;
    }
    
    return convert(signal_buffer_, buffer_size_);
}

void SignalToBinaryConverter::reset() {
    buffer_size_ = 0;
    std::memset(signal_buffer_, 0, sizeof(signal_buffer_));
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        quantile_thresholds_[i] = 0.0f;
    }
}

BinaryState SignalToBinaryConverter::applyD1Mapping(const float* signal, size_t length) {
    BinaryState state{};
    
    if (length < 2) {
        return state;
    }
    
    // Calculate derivatives over staggered micro-windows
    const size_t window_step = (length > STATE_SIZE) ? (length / STATE_SIZE) : 1;
    
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        size_t start_idx = i * window_step;
        size_t end_idx = (start_idx + window_step + 1 < length) ? start_idx + window_step + 1 : length;
        
        if (end_idx > start_idx + 1) {
            float derivative = signal[end_idx - 1] - signal[start_idx];
            state[i] = (derivative >= 0.0f);
        }
    }
    
    return state;
}

BinaryState SignalToBinaryConverter::applyD2Mapping(const float* signal, size_t length) {
    BinaryState state{};
    
    if (length == 0) {
        return state;
    }
    
    // Create sorted copy for quantile calculation
    float sorted_signal[1024];
    size_t copy_length = (length > 1024) ? 1024 : length;
    
    for (size_t i = 0; i < copy_length; ++i) {
        sorted_signal[i] = signal[i];
    }
    
    quickSort(sorted_signal, 0, static_cast<int>(copy_length - 1));
    
    // Calculate 64 quantile thresholds
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        float quantile = static_cast<float>(i + 1) / (STATE_SIZE + 1);
        size_t idx = static_cast<size_t>(quantile * (copy_length - 1));
        quantile_thresholds_[i] = sorted_signal[idx];
    }
    
    // Apply thresholds to most recent data point
    float current_value = signal[length - 1];
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        state[i] = (current_value > quantile_thresholds_[i]);
    }
    
    return state;
}

BinaryState SignalToBinaryConverter::applyD3Mapping(const float* signal, size_t length) {
    BinaryState state{};
    
    if (length < 16) {
        return state;
    }
    
    // 8 bands × 8 features approach
    constexpr size_t num_bands = 8;
    constexpr size_t features_per_band = 8;
    
    const size_t band_size = length / num_bands;
    
    for (size_t band = 0; band < num_bands; ++band) {
        size_t band_start = band * band_size;
        size_t band_end = ((band + 1) * band_size < length) ? (band + 1) * band_size : length;
        
        if (band_end <= band_start) continue;
        
        // Calculate band median
        float band_data[1024];
        size_t band_length = band_end - band_start;
        if (band_length > 1024) band_length = 1024;
        
        for (size_t i = 0; i < band_length; ++i) {
            band_data[i] = signal[band_start + i];
        }
        
        quickSort(band_data, 0, static_cast<int>(band_length - 1));
        float median = band_data[band_length / 2];
        
        // Calculate band energy
        float energy = 0.0f;
        for (size_t i = 0; i < band_length; ++i) {
            float val = signal[band_start + i];
            energy += val * val;
        }
        energy /= band_length;
        
        // Generate 8 features per band
        for (size_t feat = 0; feat < features_per_band; ++feat) {
            size_t bit_idx = band * features_per_band + feat;
            if (bit_idx >= STATE_SIZE) break;
            
            // Different feature types
            switch (feat) {
                case 0: state[bit_idx] = (energy > median); break;
                case 1: state[bit_idx] = (signal[band_start] > median); break;
                case 2: state[bit_idx] = (signal[band_end - 1] > median); break;
                case 3: state[bit_idx] = (energy > 2 * median); break;
                default: {
                    float threshold = median * (1.0f + 0.1f * feat);
                    state[bit_idx] = (energy > threshold);
                }
            }
        }
    }
    
    return state;
}

// SEPMetricsCalculator Implementation
SEPMetricsCalculator::SEPMetricsCalculator(const EMAConfig& config)
    : ema_config_(config), stability_ema_(0.0f), initialized_(false) {
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        bit_probabilities_[i] = 0.5f;
    }
}

SEPMetrics SEPMetricsCalculator::calculateMetrics(const BinaryState& prev_state, const BinaryState& curr_state) {
    SEPMetrics metrics;
    
    // Calculate overlap and flips
    BinaryState overlap_state = bitwiseAnd(prev_state, curr_state);
    BinaryState flip_state = bitwiseXor(prev_state, curr_state);
    
    metrics.overlap = popcount(overlap_state);
    metrics.flips = popcount(flip_state);
    metrics.flip_rate = static_cast<float>(metrics.flips) / STATE_SIZE;
    
    // Calculate coherence
    uint32_t n_prev = popcount(prev_state);
    uint32_t n_curr = popcount(curr_state);
    metrics.coherence = calculateCoherence(metrics.overlap, n_prev, n_curr);
    
    // Update stability
    metrics.stability = updateStability(metrics.flip_rate);
    
    // Calculate entropy
    metrics.entropy = calculateEntropy(curr_state);
    
    return metrics;
}

float SEPMetricsCalculator::calculateCoherence(uint32_t overlap, uint32_t n_prev, uint32_t n_curr) {
    if (n_prev == 0 || n_curr == 0) {
        return 0.0f;
    }
    
    // Expected overlap: E[O] = n_{t-1} * n_t / 64
    float expected_overlap = static_cast<float>(n_prev * n_curr) / STATE_SIZE;
    
    // Baseline-corrected coherence: C_t = (O_t - E[O]) / (64 - E[O])
    float denominator = STATE_SIZE - expected_overlap;
    if (denominator <= 0.0f) {
        return 0.0f;
    }
    
    float coherence = (static_cast<float>(overlap) - expected_overlap) / denominator;
    return clamp(coherence, 0.0f, 1.0f);
}

float SEPMetricsCalculator::updateStability(float flip_rate) {
    if (!initialized_) {
        stability_ema_ = 1.0f - flip_rate;
        initialized_ = true;
    } else {
        // EMA update: EMA_β[f_t] = β * f_t + (1-β) * EMA_{t-1}
        float flip_ema = ema_config_.beta * flip_rate + (1.0f - ema_config_.beta) * (1.0f - stability_ema_);
        stability_ema_ = 1.0f - flip_ema;
    }
    
    return clamp(stability_ema_, 0.0f, 1.0f);
}

float SEPMetricsCalculator::calculateEntropy(const BinaryState& state) {
    // Update bit probabilities with EMA
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        float bit_value = state[i] ? 1.0f : 0.0f;
        bit_probabilities_[i] = ema_config_.beta * bit_value + (1.0f - ema_config_.beta) * bit_probabilities_[i];
    }
    
    // Calculate Shannon entropy
    float entropy = 0.0f;
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        float p = bit_probabilities_[i];
        if (p > 0.0f && p < 1.0f) {
            entropy += -p * std::log2(p) - (1.0f - p) * std::log2(1.0f - p);
        }
    }
    
    entropy /= STATE_SIZE; // Normalize by number of bits
    return clamp(entropy, 0.0f, 1.0f);
}

void SEPMetricsCalculator::reset() {
    stability_ema_ = 0.0f;
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        bit_probabilities_[i] = 0.5f;
    }
    initialized_ = false;
}

void SEPMetricsCalculator::setEMAConfig(const EMAConfig& config) {
    ema_config_ = config;
}

uint32_t SEPMetricsCalculator::popcount(const BinaryState& state) {
    uint32_t count = 0;
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        if (state[i]) ++count;
    }
    return count;
}

BinaryState SEPMetricsCalculator::bitwiseAnd(const BinaryState& a, const BinaryState& b) {
    BinaryState result;
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        result[i] = a[i] && b[i];
    }
    return result;
}

BinaryState SEPMetricsCalculator::bitwiseXor(const BinaryState& a, const BinaryState& b) {
    BinaryState result;
    for (size_t i = 0; i < STATE_SIZE; ++i) {
        result[i] = a[i] != b[i];
    }
    return result;
}

// TriadAnalyzer Implementation
void TriadAnalyzer::computeTrajectory(
    const float* signal,
    size_t signal_length,
    SignalToBinaryConverter& converter,
    SEPMetricsCalculator& calculator,
    TriadPoint* trajectory,
    size_t* trajectory_length) {
    
    if (signal == nullptr || signal_length < 2 || trajectory == nullptr) {
        *trajectory_length = 0;
        return;
    }
    
    BinaryState prev_state{};
    size_t traj_idx = 0;
    
    for (size_t i = 1; i < signal_length && traj_idx < *trajectory_length; ++i) {
        // Convert current signal point to binary state
        BinaryState curr_state = converter.convert(signal, i + 1);
        
        // Calculate metrics
        SEPMetrics metrics = calculator.calculateMetrics(prev_state, curr_state);
        
        // Create triad point
        trajectory[traj_idx] = TriadPoint(metrics.entropy, metrics.coherence, metrics.stability, static_cast<double>(i));
        
        ++traj_idx;
        prev_state = curr_state;
    }
    
    *trajectory_length = traj_idx;
}

float TriadAnalyzer::alignTrajectories(
    const TriadPoint* trajectory1,
    size_t length1,
    const TriadPoint* trajectory2,
    size_t length2,
    float scale_factor) {
    
    if (trajectory1 == nullptr || trajectory2 == nullptr || length1 == 0 || length2 == 0) {
        return std::numeric_limits<float>::infinity();
    }
    
    // Create scaled trajectory2
    TriadPoint scaled_traj2[1024];
    size_t scaled_length = (length2 > 1024) ? 1024 : length2;
    
    timeScale(trajectory2, scaled_length, scale_factor, scaled_traj2);
    
    return calculateRMSE(trajectory1, length1, scaled_traj2, scaled_length);
}

float TriadAnalyzer::calculateRMSE(
    const TriadPoint* traj1,
    size_t length1,
    const TriadPoint* traj2,
    size_t length2) {
    
    if (traj1 == nullptr || traj2 == nullptr || length1 == 0 || length2 == 0) {
        return std::numeric_limits<float>::infinity();
    }
    
    size_t min_length = (length1 < length2) ? length1 : length2;
    float sum_squared_error = 0.0f;
    
    for (size_t i = 0; i < min_length; ++i) {
        float h_diff = traj1[i].entropy - traj2[i].entropy;
        float c_diff = traj1[i].coherence - traj2[i].coherence;
        float s_diff = traj1[i].stability - traj2[i].stability;
        
        sum_squared_error += h_diff * h_diff + c_diff * c_diff + s_diff * s_diff;
    }
    
    return std::sqrt(sum_squared_error / min_length);
}

void TriadAnalyzer::timeScale(
    const TriadPoint* input_trajectory,
    size_t input_length,
    float scale_factor,
    TriadPoint* output_trajectory) {
    
    if (input_trajectory == nullptr || output_trajectory == nullptr) {
        return;
    }
    
    for (size_t i = 0; i < input_length; ++i) {
        output_trajectory[i] = input_trajectory[i];
        output_trajectory[i].timestamp *= scale_factor;
    }
}

// StatisticalUtils Implementation
StatisticalUtils::BootstrapResult StatisticalUtils::bootstrapCI(
    const float* data,
    size_t data_length,
    size_t n_bootstrap,
    float confidence) {
    
    BootstrapResult result = {0.0f, 0.0f, 0.0f};
    
    if (data == nullptr || data_length == 0) {
        return result;
    }
    
    srand(42);  // Use fixed seed for reproducibility
    
    float bootstrap_medians[10000];
    size_t actual_bootstrap = (n_bootstrap > 10000) ? 10000 : n_bootstrap;
    
    for (size_t i = 0; i < actual_bootstrap; ++i) {
        float sample[1024];
        size_t sample_size = (data_length > 1024) ? 1024 : data_length;
        
        for (size_t j = 0; j < sample_size; ++j) {
            size_t idx = rand() % data_length;
            sample[j] = data[idx];
        }
        
        quickSort(sample, 0, static_cast<int>(sample_size - 1));
        bootstrap_medians[i] = sample[sample_size / 2];
    }
    
    quickSort(bootstrap_medians, 0, static_cast<int>(actual_bootstrap - 1));
    
    float alpha = 1.0f - confidence;
    size_t lower_idx = static_cast<size_t>(alpha / 2.0f * actual_bootstrap);
    size_t upper_idx = static_cast<size_t>((1.0f - alpha / 2.0f) * actual_bootstrap);
    size_t median_idx = actual_bootstrap / 2;
    
    result.lower_bound = bootstrap_medians[lower_idx];
    result.median = bootstrap_medians[median_idx];
    result.upper_bound = bootstrap_medians[upper_idx];
    
    return result;
}

float StatisticalUtils::wilcoxonTest(
    const float* group1,
    const float* group2,
    size_t group_size) {
    
    if (group1 == nullptr || group2 == nullptr || group_size == 0) {
        return 1.0f; // No significant difference
    }
    
    float differences[1024];
    size_t diff_count = 0;
    size_t max_diffs = (group_size > 1024) ? 1024 : group_size;
    
    for (size_t i = 0; i < max_diffs; ++i) {
        float diff = group1[i] - group2[i];
        if (std::abs(diff) > 1e-10f) {
            differences[diff_count++] = diff;
        }
    }
    
    if (diff_count == 0) {
        return 1.0f;
    }
    
    // Simplified Wilcoxon test (returns approximate p-value)
    float positive_count = 0.0f;
    for (size_t i = 0; i < diff_count; ++i) {
        if (differences[i] > 0) positive_count += 1.0f;
    }
    
    float p_approx = 2.0f * std::abs(positive_count / diff_count - 0.5f);
    return clamp(p_approx, 0.001f, 1.0f);
}

float StatisticalUtils::dtwDistance(
    const TriadPoint* traj1,
    size_t length1,
    const TriadPoint* traj2,
    size_t length2) {
    
    if (traj1 == nullptr || traj2 == nullptr || length1 == 0 || length2 == 0) {
        return std::numeric_limits<float>::infinity();
    }
    
    // Simplified DTW for small trajectories (max 64x64)
    if (length1 > 64 || length2 > 64) {
        return TriadAnalyzer::calculateRMSE(traj1, length1, traj2, length2); // Fallback to RMSE
    }
    
    float dtw[65][65];
    
    // Initialize
    for (size_t i = 0; i <= length1; ++i) {
        for (size_t j = 0; j <= length2; ++j) {
            dtw[i][j] = std::numeric_limits<float>::infinity();
        }
    }
    dtw[0][0] = 0.0f;
    
    // Fill DTW matrix
    for (size_t i = 1; i <= length1; ++i) {
        for (size_t j = 1; j <= length2; ++j) {
            // Euclidean distance between triad points
            float h_diff = traj1[i-1].entropy - traj2[j-1].entropy;
            float c_diff = traj1[i-1].coherence - traj2[j-1].coherence;
            float s_diff = traj1[i-1].stability - traj2[j-1].stability;
            float cost = std::sqrt(h_diff * h_diff + c_diff * c_diff + s_diff * s_diff);
            
            float min_prev = dtw[i-1][j];
            if (dtw[i][j-1] < min_prev) min_prev = dtw[i][j-1];
            if (dtw[i-1][j-1] < min_prev) min_prev = dtw[i-1][j-1];
            
            dtw[i][j] = cost + min_prev;
        }
    }
    
    return dtw[length1][length2];
}

void StatisticalUtils::benjaminiHochbergCorrection(
    const float* p_values,
    size_t num_p_values,
    bool* significant,
    float q_value) {
    
    if (p_values == nullptr || significant == nullptr || num_p_values == 0) {
        return;
    }
    
    // Create indexed p-values for sorting
    struct IndexedPValue {
        float p_value;
        size_t original_index;
    };
    
    IndexedPValue indexed_p_values[1024];
    size_t max_values = (num_p_values > 1024) ? 1024 : num_p_values;
    
    for (size_t i = 0; i < max_values; ++i) {
        indexed_p_values[i].p_value = p_values[i];
        indexed_p_values[i].original_index = i;
        significant[i] = false;
    }
    
    // Sort by p-value (simple bubble sort for small arrays)
    for (size_t i = 0; i < max_values - 1; ++i) {
        for (size_t j = 0; j < max_values - i - 1; ++j) {
            if (indexed_p_values[j].p_value > indexed_p_values[j + 1].p_value) {
                IndexedPValue temp = indexed_p_values[j];
                indexed_p_values[j] = indexed_p_values[j + 1];
                indexed_p_values[j + 1] = temp;
            }
        }
    }
    
    // Apply Benjamini-Hochberg correction
    for (int i = static_cast<int>(max_values) - 1; i >= 0; --i) {
        float p_i = indexed_p_values[i].p_value;
        size_t orig_idx = indexed_p_values[i].original_index;
        float threshold = (static_cast<float>(i + 1) / static_cast<float>(max_values)) * q_value;
        
        if (p_i <= threshold) {
            // Mark this and all remaining (smaller) p-values as significant
            for (int j = 0; j <= i; ++j) {
                significant[indexed_p_values[j].original_index] = true;
            }
            break;
        }
    }
}

} // namespace sep::physics_validation
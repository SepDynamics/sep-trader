#pragma once

namespace sep {
namespace core {

// Standard forex pairs used throughout the SEP trading system
inline const char* STANDARD_PAIRS[] = {
    "EUR_USD",  // Euro/US Dollar
    "GBP_USD",  // British Pound/US Dollar
    "USD_JPY",  // US Dollar/Japanese Yen
    "USD_CHF",  // US Dollar/Swiss Franc
    "AUD_USD",  // Australian Dollar/US Dollar
    "USD_CAD",  // US Dollar/Canadian Dollar
    "NZD_USD",  // New Zealand Dollar/US Dollar
    "EUR_GBP",  // Euro/British Pound
    "EUR_JPY",  // Euro/Japanese Yen
    "GBP_JPY",  // British Pound/Japanese Yen
    "AUD_JPY",  // Australian Dollar/Japanese Yen
    "EUR_CHF"   // Euro/Swiss Franc
};
inline const int STANDARD_PAIRS_COUNT = 12;

// Major forex pairs (most liquid)
inline const char* MAJOR_PAIRS[] = {
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD"
};
inline const int MAJOR_PAIRS_COUNT = 7;

// Default training pairs for quantum optimization
inline const char* DEFAULT_TRAINING_PAIRS[] = {
    "EUR_USD", "GBP_USD", "USD_JPY",
    "AUD_USD", "USD_CHF", "USD_CAD"
};
inline const int DEFAULT_TRAINING_PAIRS_COUNT = 6;

// EUR correlated pairs for multi-asset analysis
inline const char* EUR_USD_CORRELATED_PAIRS[] = {
    "GBP_USD",   // Strong positive correlation
    "AUD_USD",   // Moderate positive correlation
    "USD_CHF",   // Strong negative correlation
    "USD_JPY",   // Moderate negative correlation
    "EUR_GBP"    // Cross-pair correlation
};
inline const int EUR_USD_CORRELATED_PAIRS_COUNT = 5;

// Get pip value for a given currency pair (C-string version)
inline double getPipValue(const char* pair) {
    // Compare using C string comparison
    auto isJPYPair = [](const char* p) {
        return (p[0] == 'U' && p[1] == 'S' && p[2] == 'D' && p[4] == 'J' && p[5] == 'P' && p[6] == 'Y') ||
               (p[0] == 'E' && p[1] == 'U' && p[2] == 'R' && p[4] == 'J' && p[5] == 'P' && p[6] == 'Y') ||
               (p[0] == 'G' && p[1] == 'B' && p[2] == 'P' && p[4] == 'J' && p[5] == 'P' && p[6] == 'Y') ||
               (p[0] == 'A' && p[1] == 'U' && p[2] == 'D' && p[4] == 'J' && p[5] == 'P' && p[6] == 'Y');
    };
    
    if (isJPYPair(pair)) {
        return 0.01;  // JPY pairs have 2 decimal places
    }
    return 0.0001;  // Most pairs have 4 decimal places
}

// Check if a pair is a major pair (C-string version)
inline bool isMajorPair(const char* pair) {
    for (int i = 0; i < MAJOR_PAIRS_COUNT; ++i) {
        const char* major = MAJOR_PAIRS[i];
        // Simple string comparison
        bool match = true;
        for (int j = 0; major[j] != '\0' && pair[j] != '\0'; ++j) {
            if (major[j] != pair[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

} // namespace core
} // namespace sep
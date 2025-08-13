// /sep/src/common/pqxx_time_point_traits.h
#pragma once

#include <pqxx/pqxx>
#include <chrono>
#include <string>
#include <string_view>
#include <sstream>
#include <iomanip> // For std::put_time, std::get_time
#include <ctime>   // For std::tm, gmtime

namespace pqxx {
    // Full specialization for std::chrono::system_clock::time_point
    template<>
    struct string_traits<std::chrono::time_point<std::chrono::system_clock>> {
        using subject_type = std::chrono::time_point<std::chrono::system_clock>;
        
        static constexpr const char* name() noexcept { return "time_point"; }
        
        static constexpr bool has_null() noexcept { return true; }
        
        static bool is_null(const subject_type& obj) {
            // A time_point is never conceptually "null". If you use epoch as a sentinel, check here.
            return obj.time_since_epoch().count() == 0;
        }

        static subject_type null() {
            // Return epoch time as null representation
            return std::chrono::time_point<std::chrono::system_clock>{};
        }

        static std::string to_string(const subject_type& obj) {
            auto tt = std::chrono::system_clock::to_time_t(obj);
            std::tm tm = *std::gmtime(&tt); // Use gmtime for UTC
            std::stringstream ss;
            // Format as ISO 8601, which PostgreSQL loves.
            ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
            return ss.str();
        }

        // THIS IS THE CORRECTED FUNCTION
        static void from_string(const char* str, subject_type& obj) {
            if (str == nullptr || *str == '\0') {
                obj = std::chrono::system_clock::time_point{}; // Handle NULL from DB
                return;
            }
            std::tm tm{};
            std::stringstream ss{str};
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

            if (ss.fail()) {
                throw std::runtime_error("Failed to parse time_point from string: " + std::string(str));
            }
            obj = std::chrono::system_clock::from_time_t(timegm(&tm));
        }
        
        static std::size_t size_buffer(const subject_type& /*obj*/) {
            return 32; // Enough for ISO timestamp
        }
    };
} // namespace pqxx

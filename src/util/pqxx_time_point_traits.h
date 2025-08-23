// /sep/src/common/pqxx_time_point_traits.h
#pragma once

#include <chrono>
#include <string>
#include <string_view>
#include <sstream>
#include <iomanip> // For std::put_time, std::get_time
#include <ctime>   // For std::tm, gmtime

// Forward declaration of pqxx::string_traits to allow specialization prior to
// including <pqxx/pqxx>.  This avoids early instantiation of pqxx::nullness
// with incomplete constexpr flags, which previously triggered build errors.
namespace pqxx {
    template <typename T>
    struct string_traits;
}

namespace pqxx {
    // Partial specialization for std::chrono::sys_time (system_clock::time_point)
    template <typename Duration>
    struct string_traits<std::chrono::sys_time<Duration>> {
        using subject_type = std::chrono::sys_time<Duration>;

        static constexpr const char* name() noexcept { return "time_point"; }

        // Provide constexpr flags for pqxx::nullness detection
        inline static constexpr bool has_null = true;
        inline static constexpr bool always_null = false;

        static bool is_null(const subject_type& obj) {
            // A time_point is never conceptually "null". If you use epoch as a sentinel, check here.
            return obj.time_since_epoch().count() == 0;
        }

        static subject_type null() {
            // Return epoch time as null representation
            return subject_type{};
        }

        static std::string to_string(const subject_type& obj) {
            auto sys_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(obj);
            auto tt = std::chrono::system_clock::to_time_t(sys_time);
            std::tm tm = *std::gmtime(&tt); // Use gmtime for UTC
            std::stringstream ss;
            // Format as ISO 8601, which PostgreSQL loves.
            ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
            return ss.str();
        }

        static void from_string(const char* str, subject_type& obj) {
            if (str == nullptr || *str == '\0') {
                obj = subject_type{}; // Handle NULL from DB
                return;
            }
            std::tm tm{};
            std::stringstream ss{str};
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

            if (ss.fail()) {
                throw std::runtime_error("Failed to parse time_point from string: " + std::string(str));
            }
            auto sys_time = std::chrono::system_clock::from_time_t(timegm(&tm));
            obj = std::chrono::time_point_cast<Duration>(sys_time);
        }

        static std::size_t size_buffer(const subject_type& /*obj*/) {
            return 32; // Enough for ISO timestamp
        }
    };
} // namespace pqxx

#include <pqxx/pqxx>

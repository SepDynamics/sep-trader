#pragma once

#include <pqxx/pqxx>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace pqxx {
    // Specialization for std::chrono::system_clock::time_point
    template<>
    struct string_traits<std::chrono::time_point<std::chrono::system_clock>> {
        using time_point_type = std::chrono::time_point<std::chrono::system_clock>;
        
        static bool is_null(const time_point_type&) {
            return false; // time_point instances are never null conceptually
        }

        static std::string to_string(const time_point_type& obj) {
            // Convert to time_t first, then to tm struct (UTC)
            auto tt = std::chrono::system_clock::to_time_t(obj);
            std::tm tm = *std::gmtime(&tt); // Using gmtime for UTC

            std::stringstream ss;
            // Format as ISO 8601, which PostgreSQL can easily parse
            ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S"); 
            return ss.str();
        }

        static void from_string(const char* str, time_point_type& obj) {
            if (str == nullptr || *str == '\0') {
                // Handle null/empty string
                obj = time_point_type{};
                return;
            }
            
            std::tm tm{};
            std::stringstream ss(str);
            // Parse from ISO 8601 format
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S"); 

            if (ss.fail()) {
                throw std::runtime_error("Failed to parse time_point from string: " + std::string(str));
            }
            
            // Convert tm to time_t (mktime assumes local time, but we'll use it for now)
            // For precise UTC handling, a more robust solution would be needed
            obj = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        }
        
        // REQUIRED: null() method for pqxx to handle NULL database values
        static std::string null() {
            return ""; // Empty string represents NULL for timestamp in PostgreSQL
        }
    };
} // namespace pqxx

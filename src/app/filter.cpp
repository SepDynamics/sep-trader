#include "filter.h"
#include "util/financial_data_types.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sep {
namespace services {

namespace {
std::string trim(std::string_view sv) {
    size_t start = 0;
    size_t end = sv.size();
    while (start < end && std::isspace(static_cast<unsigned char>(sv[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(sv[end - 1]))) --end;
    return std::string(sv.substr(start, end - start));
}

std::chrono::time_point<std::chrono::system_clock> parse_time(const std::string& s) {
    std::string ts = s;
    if (ts.find('T') == std::string::npos) {
        ts += "T00:00:00.000000Z";
    } else if (ts.back() != 'Z') {
        if (ts.find('.') == std::string::npos) ts += ".000000";
        ts += 'Z';
    }
    return common::parseTimestamp(ts);
}

bool wildcard_match(const std::string& value, const std::string& pattern) {
    if (pattern.size() >= 1 && pattern.front() == '*' && pattern.back() == '*') {
        return value.find(pattern.substr(1, pattern.size() - 2)) != std::string::npos;
    } else if (!pattern.empty() && pattern.front() == '*') {
        auto suffix = pattern.substr(1);
        if (value.size() < suffix.size()) return false;
        return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    } else if (!pattern.empty() && pattern.back() == '*') {
        auto prefix = pattern.substr(0, pattern.size() - 1);
        return value.rfind(prefix, 0) == 0;
    }
    return value == pattern;
}

} // namespace

Filter parse_filter(std::string_view expr) {
    Filter f;
    size_t pos = 0;
    std::string s(expr);
    while (pos < s.size()) {
        size_t next = s.find("&&", pos);
        std::string cond = trim(s.substr(pos, next == std::string::npos ? std::string::npos : next - pos));
        if (!cond.empty()) {
            std::string op;
            size_t op_pos = std::string::npos;
            const std::vector<std::string> ops = {"==", ">=", "<=", ">", "<", "~=", "="};
            for (const auto& candidate : ops) {
                op_pos = cond.find(candidate);
                if (op_pos != std::string::npos) {
                    op = candidate;
                    break;
                }
            }
            if (op.empty()) {
                std::cerr << "[filter] invalid query" << std::endl;
                throw std::runtime_error("Invalid filter");
            }
            std::string field = trim(cond.substr(0, op_pos));
            std::string value = trim(cond.substr(op_pos + op.size()));

            if (field == "pair" || field == "id") {
                if (op == "==") {
                    std::string cmp = value;
                    f.predicates.push_back([field, cmp](const Row& r) {
                        const std::string& val = (field == "pair") ? r.pair : r.id;
                        return val == cmp;
                    });
                } else if (op == "~=") {
                    std::string pattern = value;
                    f.predicates.push_back([field, pattern](const Row& r) {
                        const std::string& val = (field == "pair") ? r.pair : r.id;
                        return wildcard_match(val, pattern);
                    });
                } else {
                    std::cerr << "[filter] invalid query" << std::endl;
                    throw std::runtime_error("Invalid filter");
                }
            } else if (field == "score") {
                if (op == "==" || op == ">" || op == ">=" || op == "<" || op == "<=") {
                    try {
                        double cmp = std::stod(value);
                        if (op == "==")
                            f.predicates.push_back([cmp](const Row& r) { return r.score == cmp; });
                        else if (op == ">")
                            f.predicates.push_back([cmp](const Row& r) { return r.score > cmp; });
                        else if (op == ">=")
                            f.predicates.push_back([cmp](const Row& r) { return r.score >= cmp; });
                        else if (op == "<")
                            f.predicates.push_back([cmp](const Row& r) { return r.score < cmp; });
                        else if (op == "<=")
                            f.predicates.push_back([cmp](const Row& r) { return r.score <= cmp; });
                    } catch (const std::exception&) {
                        std::cerr << "[filter] invalid query" << std::endl;
                        throw std::runtime_error("Invalid filter");
                    }
                } else if (op == "=") {
                    if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
                        std::cerr << "[filter] invalid query" << std::endl;
                        throw std::runtime_error("Invalid filter");
                    }
                    auto comma = value.find(',');
                    if (comma == std::string::npos) {
                        std::cerr << "[filter] invalid query" << std::endl;
                        throw std::runtime_error("Invalid filter");
                    }
                    try {
                        double a = std::stod(trim(value.substr(1, comma - 1)));
                        double b = std::stod(trim(value.substr(comma + 1, value.size() - comma - 2)));
                        f.predicates.push_back([a, b](const Row& r) { return r.score >= a && r.score <= b; });
                    } catch (const std::exception&) {
                        std::cerr << "[filter] invalid query" << std::endl;
                        throw std::runtime_error("Invalid filter");
                    }
                } else {
                    std::cerr << "[filter] invalid query" << std::endl;
                    throw std::runtime_error("Invalid filter");
                }
            } else if (field == "ts") {
                try {
                    if (op == "==" || op == ">" || op == ">=" || op == "<" || op == "<=") {
                        auto cmp = parse_time(value);
                        if (op == "==")
                            f.predicates.push_back([cmp](const Row& r) { return r.ts == cmp; });
                        else if (op == ">")
                            f.predicates.push_back([cmp](const Row& r) { return r.ts > cmp; });
                        else if (op == ">=")
                            f.predicates.push_back([cmp](const Row& r) { return r.ts >= cmp; });
                        else if (op == "<")
                            f.predicates.push_back([cmp](const Row& r) { return r.ts < cmp; });
                        else if (op == "<=")
                            f.predicates.push_back([cmp](const Row& r) { return r.ts <= cmp; });
                    } else if (op == "=") {
                        if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
                            std::cerr << "[filter] invalid query" << std::endl;
                            throw std::runtime_error("Invalid filter");
                        }
                        auto comma = value.find(',');
                        if (comma == std::string::npos) {
                            std::cerr << "[filter] invalid query" << std::endl;
                            throw std::runtime_error("Invalid filter");
                        }
                        auto a = parse_time(trim(value.substr(1, comma - 1)));
                        auto b = parse_time(trim(value.substr(comma + 1, value.size() - comma - 2)));
                        f.predicates.push_back([a, b](const Row& r) { return r.ts >= a && r.ts <= b; });
                    } else {
                        std::cerr << "[filter] invalid query" << std::endl;
                        throw std::runtime_error("Invalid filter");
                    }
                } catch (const std::exception&) {
                    std::cerr << "[filter] invalid query" << std::endl;
                    throw std::runtime_error("Invalid filter");
                }
            } else {
                std::cerr << "[filter] invalid query" << std::endl;
                throw std::runtime_error("Invalid filter");
            }
        }
        if (next == std::string::npos) break;
        pos = next + 2;
    }
    return f;
}

bool match(const Row& row, const Filter& filter) {
    for (const auto& p : filter.predicates) {
        if (!p(row)) return false;
    }
    return true;
}

} // namespace services
} // namespace sep

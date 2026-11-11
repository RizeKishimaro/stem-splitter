// A minimal test runner so the shared code builds and tests with plain CMake
// and no third-party dependency. Each TEST registers itself; the runner takes
// an optional name prefix to run one group, which CTest uses.
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace soniclab::test {

struct TestCase {
    std::string name;
    std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Failure {
    std::string message;
};

struct Registrar {
    Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

inline void fail(const char* file, int line, const std::string& what) {
    std::ostringstream out;
    out << file << ':' << line << ": " << what;
    throw Failure{out.str()};
}

}  // namespace soniclab::test

#define SL_CONCAT_INNER(a, b) a##b
#define SL_CONCAT(a, b) SL_CONCAT_INNER(a, b)

#define TEST(name)                                                                              \
    static void SL_CONCAT(test_body_, __LINE__)();                                              \
    static ::soniclab::test::Registrar SL_CONCAT(test_registrar_, __LINE__)(name,               \
                                                                         &SL_CONCAT(test_body_, __LINE__)); \
    static void SL_CONCAT(test_body_, __LINE__)()

#define CHECK(condition)                                                                   \
    do {                                                                                   \
        if (!(condition)) ::soniclab::test::fail(__FILE__, __LINE__, "CHECK(" #condition ")"); \
    } while (false)

#define CHECK_EQ(actual, expected)                                                          \
    do {                                                                                    \
        const auto sl_a = (actual); /* copies: a reference could dangle */                 \
        const auto sl_e = (expected);                                                       \
        if (!(sl_a == sl_e)) {                                                              \
            std::ostringstream sl_out;                                                      \
            sl_out << "CHECK_EQ(" #actual ", " #expected "): " << sl_a << " != " << sl_e;   \
            ::soniclab::test::fail(__FILE__, __LINE__, sl_out.str());                       \
        }                                                                                   \
    } while (false)

#define CHECK_NEAR(actual, expected, tolerance)                                                       \
    do {                                                                                              \
        const double sl_a = (actual);                                                                 \
        const double sl_e = (expected);                                                               \
        if (std::fabs(sl_a - sl_e) > (tolerance)) {                                                   \
            std::ostringstream sl_out;                                                                \
            sl_out << "CHECK_NEAR(" #actual ", " #expected "): " << sl_a << " vs " << sl_e;          \
            ::soniclab::test::fail(__FILE__, __LINE__, sl_out.str());                                 \
        }                                                                                             \
    } while (false)

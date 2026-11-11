#include <cstring>

#include "test_framework.h"

int main(int argc, char** argv) {
    const char* prefix = argc > 1 ? argv[1] : "";
    int run = 0;
    int failed = 0;
    for (const auto& test : soniclab::test::registry()) {
        if (std::strncmp(test.name.c_str(), prefix, std::strlen(prefix)) != 0) continue;
        ++run;
        try {
            test.body();
            std::cout << "[ ok ] " << test.name << '\n';
        } catch (const soniclab::test::Failure& failure) {
            ++failed;
            std::cout << "[FAIL] " << test.name << "\n       " << failure.message << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cout << "[FAIL] " << test.name << "\n       exception: " << error.what() << '\n';
        }
    }
    std::cout << run << " tests, " << failed << " failed\n";
    if (run == 0) {
        std::cout << "no tests match '" << prefix << "'\n";
        return 1;
    }
    return failed == 0 ? 0 : 1;
}

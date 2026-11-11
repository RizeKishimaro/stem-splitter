#include "soniclab/inference/tensor.h"

#include <limits>
#include <sstream>

namespace soniclab::inference {

bool elementCount(const std::vector<std::int64_t>& shape, std::size_t& count) {
    std::size_t total = 1;
    for (const auto dim : shape) {
        if (dim < 0) return false;
        const auto size = static_cast<std::size_t>(dim);
        if (size != 0 && total > std::numeric_limits<std::size_t>::max() / size) return false;
        total *= size;
    }
    count = total;
    return true;
}

bool shapeMatches(const std::vector<std::int64_t>& expected, const std::vector<std::int64_t>& actual) {
    if (expected.size() != actual.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] >= 0 && expected[i] != actual[i]) return false;
        if (actual[i] < 0) return false;
    }
    return true;
}

std::string shapeText(const std::vector<std::int64_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

bool Tensor::assign(std::vector<std::int64_t> shape, std::vector<float> values) {
    std::size_t count = 0;
    if (!elementCount(shape, count) || count != values.size()) return false;
    shape_ = std::move(shape);
    values_ = std::move(values);
    return true;
}

}  // namespace soniclab::inference

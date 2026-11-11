// Contiguous float32 tensor. Row-major, the layout ONNX calls NCHW when the
// shape is named that way: the last dimension changes fastest.
//
// A Tensor owns its buffer. Moving it transfers that ownership. data() is
// valid until the Tensor is moved-from or destroyed. Callers that only need
// to read a result should take the Tensor by value.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace soniclab::inference {

class Tensor {
public:
    Tensor() = default;

    // Fails (returns false and leaves this unchanged) when a dimension is
    // negative or the value count is not the product of the shape.
    bool assign(std::vector<std::int64_t> shape, std::vector<float> values);

    const std::vector<std::int64_t>& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return values_.size(); }
    bool empty() const noexcept { return values_.empty(); }

    float* data() noexcept { return values_.data(); }
    const float* data() const noexcept { return values_.data(); }

    const std::vector<float>& values() const noexcept { return values_; }

private:
    std::vector<std::int64_t> shape_;
    std::vector<float> values_;
};

// Product of a fully static shape. A negative dimension is dynamic and
// makes the count unknown. Overflow returns false.
bool elementCount(const std::vector<std::int64_t>& shape, std::size_t& count);

// True when ``actual`` can be fed to a model input of ``expected``.
// A negative expected dimension accepts any non-negative actual dimension.
bool shapeMatches(const std::vector<std::int64_t>& expected, const std::vector<std::int64_t>& actual);

std::string shapeText(const std::vector<std::int64_t>& shape);

}  // namespace soniclab::inference

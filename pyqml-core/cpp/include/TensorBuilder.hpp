#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <pyqml-core/cpp/include/VectorRef.hpp>

enum class DType;
enum class Device;
class Tensor;

struct OperandInfo {
    void* data = nullptr;

    // Byte strides, one per dimension.
    std::vector<std::int64_t> strides;

    DType dtype;
    Device device;

    bool is_output = false;
    bool is_read_write = false;
};

class TensorBuilder {
public:
    TensorBuilder() = default;

    void add_input(Tensor& tensor);
    void add_output(Tensor& tensor);

    // Validate operands and prepare iteration metadata.
    void build();
    void reset();

    // Shape and operand information.
    std::size_t ndim() const noexcept;
    std::size_t numel() const noexcept;
    std::size_t operand_count() const noexcept;
    std::size_t output_count() const noexcept;

    const std::vector<std::size_t>& shape() const noexcept;

    const OperandInfo& operand(std::size_t index) const;
    OperandInfo& operand(std::size_t index);

    DType common_dtype() const noexcept;
    Device common_device() const noexcept;

    // Iteration metadata construction.
    void compute_output_shape();
    void compute_broadcast_strides();

    void promote_common_dtype();
    void promote_common_device();

    bool is_contiguous() const noexcept;
    bool is_trivial_1d() const noexcept;

private:
    VectorRef<OperandInfo, 4> operands_;
    std::vector<std::size_t> shape_;

    std::size_t numel_ = 0;
    std::size_t num_outputs_ = 0;

    DType common_dtype_;
    Device common_device_;

    bool built_ = false;
};

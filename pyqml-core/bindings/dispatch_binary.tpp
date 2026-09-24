#include "Tensor.hpp"
#include "dispatch.hpp"


template <typename U, typename V, typename FUNC>
Tensor op_Binary(const Tensor &a, const Tensor &b, FUNC &&op)
{
    auto a_tens = a.get_typed_tensor<U>();
    auto b_tens = b.get_typed_tensor<V>();
    DType result = (static_cast<int>(a.type()) > static_cast<int>(b.type())) ? a.type() : b.type();
    auto tens = op(a_tens, b_tens);
    return Tensor(tens.owner(), tens.dim(), result);
}


template <typename Op>
struct Binary_Dispatch_Table
{
    using Binary_Dispatch_Func = Tensor (*)(const Tensor &, const Tensor &, Op &&);
    using arr = std::array<std::array<Binary_Dispatch_Func, 6>, 6>;
    arr binary_table;

    Binary_Dispatch_Table()
    {
        REGISTER_DTYPE_ROW(binary_table, DType::Int8, int8_t, Op, op_Binary);
        REGISTER_DTYPE_ROW(binary_table, DType::Int16, int16_t, Op, op_Binary);
        REGISTER_DTYPE_ROW(binary_table, DType::Int32, int32_t, Op, op_Binary);
        REGISTER_DTYPE_ROW(binary_table, DType::Int64, int64_t, Op, op_Binary);
        REGISTER_DTYPE_ROW(binary_table, DType::Float32, float, Op, op_Binary);
        REGISTER_DTYPE_ROW(binary_table, DType::Float64, double, Op, op_Binary);
    }
};


template <typename Func>
Tensor bin_op(const Tensor &a, const Tensor &b, Func &&op)
{
    Binary_Dispatch_Table<Func> table;
    return table.binary_table[(int)a.type()][(int)b.type()](a, b, std::forward<Func>(op));
}

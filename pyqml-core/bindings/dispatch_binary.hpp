
#include "Tensor.hpp"
#include "dispatch.hpp"


#define BINARY_DISPATCH(a_tens, b_tens, ...)                                         \
    PYQ_UNARY_TENSOR_DISPATCH(a_tens, using type1 = atype;                           \
                              PYQ_UNARY_TENSOR_DISPATCH(b_tens, using type2 = atype; \
                                                        __VA_ARGS__))

#define BINARY_OP_DISPATCH(a_tens, b_tens, Op)                                                                                          \
    BINARY_DISPATCH(a_tens, b_tens,                                                                                                     \
                    auto typed_a = a_tens.get_typed_tensor<type1>();                                                                       \
                    auto typed_b = b_tens.get_typed_tensor<type2>();                                                                       \
                    DType result = (static_cast<int>(a_tens.type()) > static_cast<int>(b_tens.type())) ? a_tens.type() : b_tens.type(); \
                    auto res = Op(typed_a, typed_b);                                                                                    \
                    return Tensor(res.owner(), res.dim(), result);)

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



 template <typename Op>
    Tensor Tensor::dispatchOp(const Tensor &a, const Tensor &b, Op &&opy)
    {
        return Tensor::disp_2(a.dtype, b.dtype, [&](auto t1, auto t2)
                              {
        using T = std::decay_t<decltype(t1)>;
        using U = std::decay_t<decltype(t2)>;


        T* raw_a = static_cast<T*>(a.data.get());
        std::shared_ptr<T[]> data_a (a.data, raw_a);
        U* raw_b = static_cast<U*>(b.data.get());
        std::shared_ptr<U[]> data_b (b.data, raw_b);
        tensor<T> a_tens = tensor<T>::tensor_view(data_a, a.shape_, a.strides_, a.offset, a.size);
        tensor<U> b_tens = tensor<U>::tensor_view(data_b, b.shape_, b.strides_, b.offset, b.size);
        DType result = (static_cast<int>(a.dtype) > static_cast<int>(b.dtype)) ? a.dtype: b.dtype;
        auto tens = opy(a_tens, b_tens);
        return Tensor(tens.owner(), tens.dim(), result); });
    }


   

    // This method converts the tensor to a different dtype and optionally copies the memory,
    // enabling explicit type control when interacting with Python or native code.
    Tensor Tensor::astype(DType new_type, bool h) const
    {
        return Tensor::disp_2(dtype, new_type, [&](auto t1, auto t2)
                              {
        using T = std::decay_t<decltype(t1)>;
        using U = std::decay_t<decltype(t2)>;
        T* raw_a = static_cast<T*>(data.get());
        std::shared_ptr<T[]> data_a (data, raw_a);
        tensor<T> a_tens = tensor<T>::tensor_view(data_a, shape_, strides_, offset, size);
        tensor<U> res = a_tens.template astype<U>(h);
        return Tensor(res.owner(), res.dim(), res.strides(), new_type, res.ofst()); });
    }

 



Tensor Tensor::operator+(const Tensor &other) const
    {
        return bin_op(*this, other, [&](auto &t_1, auto &t_2)
                      { return binary_ops(t_1, t_2, std::plus<>()); });
    }

    // This overload implements elementwise subtraction between two tensors and returns the
    // result as a new tensor with the broadcasted shape of the inputs.
    Tensor Tensor::operator-(const Tensor &other) const {
        /*return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::minus<>()); });*/

        BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                           { return binary_ops(t_1, t_2, std::minus<>()); });}

    // This overload implements elementwise multiplication between two tensors and returns a
    // new tensor that reflects the broadcasted shape of the operands.
    Tensor Tensor::operator*(const Tensor &other) const
    {
        BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                           { return binary_ops(t_1, t_2, std::multiplies<>()); });}
    

    // This overload implements elementwise division between two tensors and returns the
    // quotient as a new tensor while preserving the broadcasted shape semantics.
    Tensor Tensor::operator/(const Tensor &other) const
    {
        BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                           { return binary_ops(t_1, t_2, std::divides<>()); });}
    

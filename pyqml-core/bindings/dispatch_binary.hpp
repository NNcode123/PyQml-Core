
#include "Tensor.hpp"
#include "dispatch.hpp"
#include "Autograd/attach_grad_binary.hpp"





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




   

    // This method converts the tensor to a different dtype and optionally copies the memory,
    // enabling explicit type control when interacting with Python or native code.
    Tensor Tensor::astype(DType new_type, [[maybe_unused]] bool h) const
    {

       if (dtype == new_type){
        return *this;
       }
    
       TYPE_CAST_DISPATCH((*this), new_type)
    }

 



Tensor Tensor::operator+(const Tensor &other) const
    {
        auto Tens = bin_op(*this, other, [&](auto &t_1, auto &t_2)
                      { return binary_ops(t_1, t_2, std::plus<>()); });
        return Tens;
    }

    // This overload implements elementwise subtraction between two tensors and returns the
    // result as a new tensor with the broadcasted shape of the inputs.
    Tensor Tensor::operator-(const Tensor &other) const {
        /*return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::minus<>()); });*/

        auto Tens =[this, &other]()->Tensor{ BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                           { return binary_ops(t_1, t_2, std::minus<>()); }); } ();

            return Tens;
       //Attach_Grad(Tens,Sub,(*this),other)
    }

    // This overload implements elementwise multiplication between two tensors and returns a
    // new tensor that reflects the broadcasted shape of the operands.
    Tensor Tensor::operator*(const Tensor &other) const
    {
        
        BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                           { return binary_ops(t_1, t_2, std::multiplies<>()); });
                        }

    

    // This overload implements elementwise division between two tensors and returns the
    // quotient as a new tensor while preserving the broadcasted shape semantics.
    Tensor Tensor::operator/(const Tensor &other) const
    {
        BINARY_OP_DISPATCH((*this), other, [&](auto &t_1, auto &t_2)
                           { return binary_ops(t_1, t_2, std::divides<>()); });
                        
    }
    
    Tensor& Tensor::operator+=(const Tensor& other){
        if (!data){
            *this = other;
            return *this;
        }
        *this = *this + other;
        return *this;
    }



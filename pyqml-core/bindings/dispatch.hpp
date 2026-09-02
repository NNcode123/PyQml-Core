#pragma once
#include <array>

#define PYQ_CASE(DTYPE, CTYPE, ...)\
    case DTYPE:\
    {\
        using atype = CTYPE;        \
        __VA_ARGS__                 \
    }
    

#define PYQ_SWITCH_IMPL(Type, ...)\
    switch (Type)\
    {\
        __VA_ARGS__                \
    }

#define PYQ_UNARY_DISPATCH(TYPE, BODY)\
    PYQ_SWITCH_IMPL(TYPE, PYQ_CASE(DType::Int64, int64_t, BODY) PYQ_CASE(DType::Int32, int32_t, BODY) PYQ_CASE(DType::Int16, int16_t, BODY) PYQ_CASE(DType::Int8, int8_t, BODY) PYQ_CASE(DType::Float32, float, BODY) PYQ_CASE(DType::Float64, double, BODY) /*default : { throw std::runtime_error("Unsupported DType"); }*/)

#define REGISTER_DTYPE_ROW(TABLE, DTYPE, type, OP, FUNC) \
    TABLE[(int)DTYPE] =                                  \
        {                                                \
            &FUNC<type, int8_t, OP>,                     \
            &FUNC<type, int16_t, OP>,                    \
            &FUNC<type, int32_t, OP>,                    \
            &FUNC<type, int64_t, OP>,                    \
            &FUNC<type, float, OP>,                      \
            &FUNC<type, double, OP>}\


#define BINARY_DISPATCH(a_tens, b_tens, ...)                                         \
    PYQ_UNARY_DISPATCH(a_tens.type(), using type1 = atype;                           \
                              PYQ_UNARY_DISPATCH(b_tens.type(), using type2 = atype;\
                                                        __VA_ARGS__))\

#define BINARY_OP_DISPATCH(a_tens, b_tens, Op)                                                                                          \
    BINARY_DISPATCH(a_tens, b_tens,                                                                                                     \
                    auto typed_a = a_tens.get_typed_tensor<type1>();                                                                       \
                    auto typed_b = b_tens.get_typed_tensor<type2>();                                                                       \
                    DType result = (static_cast<int>(a_tens.type()) > static_cast<int>(b_tens.type())) ? a_tens.type() : b_tens.type(); \
                    auto res = Op(typed_a, typed_b);                                                                                    \
                    return Tensor(res.owner(), res.dim(), result);)\



#define TYPE_CAST_DISPATCH(tens, TYPE)\
    PYQ_UNARY_DISPATCH(tens.type(), using type1 = atype;  \
                            PYQ_UNARY_DISPATCH(TYPE, using type2 = atype;\
                            auto typed_tens = tens.get_typed_tensor<type1>();\
                            auto convert_tens = typed_tens.astype<type2>();\
                            return Tensor(convert_tens.owner(), convert_tens.dim(), TYPE);)\
    )

#define GET_TENSOR_PROP(TENSOR, PROP)\
    PYQ_UNARY_DISPATCH(TENSOR.type(),\
    auto typed_tens = TENSOR.get_typed_tensor<atype>();\
    auto transformed_tens = PROP(typed_tens);\
    return Tensor(transformed_tens.owner(), transformed_tens.dim(), transformed_tens.strides(),TENSOR.type(), transformed_tens.ofst());\
)

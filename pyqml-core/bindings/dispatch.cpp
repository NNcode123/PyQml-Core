#include "Tensor.hpp"

#define PYQ_CASE(DTYPE, CTYPE, FUNC) \
    case DTYPE:                      \
    {                                \
        using type = CTYPE;          \
        FUNC();                      \
        break;                       \
    }

#define PYQ_SWITCH_IMPL(Type, ...) \
    switch (Type)                  \
    {                              \
        __VA_ARGS__                \
    }

#define PYQ_TENSOR_BUILD(TENSOR, DTYPE)                                                                          \
    PYQ_SWITCH_IMPL(DTYPE,                                                                                       \
                    PYQ_CASE(DType::Int64, int64_t, [&] { return Tensor::getTens<type>(TENSOR); })               \
                        PYQ_CASE(DType::Int32, int32_t, [&] { return Tensor::getTens<type>(TENSOR); })           \
                            PYQ_CASE(DType::Int16, int16_t, [&] { return Tensor::getTens<type>(TENSOR); })       \
                                PYQ_CASE(DType::Int8, int8_t, [&] { return Tensor::getTens<type>(TENSOR); })     \
                                    PYQ_CASE(DType::Float, float, [&] { return Tensor::getTens<type>(TENSOR); }) \
                                        PYQ_CASE(DType::Double, double, [&] { return Tensor::getTens<type>(TENSOR); })\
                                        default: {\
                                            throw std::runtime_error("Unsupported DType");\
                                        })

/*
template <typename T>
static tensor<T> getTens(const Tensor &t)
{
    T *raw = static_cast<T *>(data.get());
    std::shared_ptr<T[]> data_n(data, raw);
    tensor<T> tens = tensor_view<T>(data_n, shape_, strides_, offset, size);
    return tens;
}
    */

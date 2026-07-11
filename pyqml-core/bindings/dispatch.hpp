#pragma once
#include <array>

#define PYQ_CASE(DTYPE, CTYPE, ...) \
    case DTYPE:                     \
    {                               \
        using atype = CTYPE;        \
        __VA_ARGS__                 \
    }

#define PYQ_SWITCH_IMPL(Type, ...) \
    switch (Type)                  \
    {                              \
        __VA_ARGS__                \
    }

#define PYQ_UNARY_DISPATCH(TYPE, BODY) \
    PYQ_SWITCH_IMPL(TYPE, PYQ_CASE(DType::Int64, int64_t, BODY) PYQ_CASE(DType::Int32, int32_t, BODY) PYQ_CASE(DType::Int16, int16_t, BODY) PYQ_CASE(DType::Int8, int8_t, BODY) PYQ_CASE(DType::Float32, float, BODY) PYQ_CASE(DType::Float64, double, BODY) default : { throw std::runtime_error("Unsupported DType"); })

#define REGISTER_DTYPE_ROW(TABLE, DTYPE, type, OP, FUNC) \
    TABLE[(int)DTYPE] =                                  \
        {                                                \
            &FUNC<type, int8_t, OP>,                     \
            &FUNC<type, int16_t, OP>,                    \
            &FUNC<type, int32_t, OP>,                    \
            &FUNC<type, int64_t, OP>,                    \
            &FUNC<type, float, OP>,                      \
            &FUNC<type, double, OP>}

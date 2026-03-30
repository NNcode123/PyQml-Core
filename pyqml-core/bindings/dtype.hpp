#pragma once
#include <variant>
#include <tuple>
#include <cmath>
#include "bindings.hpp"
#include "../cpp/src/tensor.hpp"

enum class DType
{
    NoneType,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
};

enum class OpType
{

    Add,
    Div,
    Mul,
    Sub,
    MatMul,
    TensorProd,
    InnerProd,
    concat
};

/*
struct DispatchTable{
    template <typename ... Types>
    using Type_list = std::tuple<DType>;
    using Func = (Tensor *)op(Tensor,Tensor&);

    Type_list q;

}
*/

template <typename T>
struct typeDType;

// Ints
template <>
struct typeDType<int8_t>
{
    static constexpr DType type = DType::Int8;
};

template <>
struct typeDType<int16_t>
{
    static constexpr DType type = DType::Int16;
};

template <>
struct typeDType<int32_t>
{
    static constexpr DType type = DType::Int32;
};

template <>
struct typeDType<int64_t>
{
    static constexpr DType type = DType::Int64;
};

// Floats
template <>
struct typeDType<float>
{
    static constexpr DType type = DType::Float32;
};

template <>
struct typeDType<double>
{
    static constexpr DType type = DType::Float64;
};

template <typename T>
struct TypeRank
{
};

template <>
struct TypeRank<int8_t>
{
    static constexpr int rank = 0;
};
template <>
struct TypeRank<int16_t>
{
    static constexpr int rank = 1;
};
template <>
struct TypeRank<int32_t>
{
    static constexpr int rank = 2;
};
template <>
struct TypeRank<int64_t>
{
    static constexpr int rank = 3;
};
/*
template <>
struct TypeRank<int>
{
    static constexpr int rank = 4;
};
*/
template <>
struct TypeRank<float>
{
    static constexpr int rank = 4;
};
template <>
struct TypeRank<double>
{
    static constexpr int rank = 5;
};

template <typename U, typename V>

struct promote
{
    static constexpr int rank =
        (TypeRank<U>::rank > TypeRank<V>::rank)
            ? TypeRank<U>::rank
            : TypeRank<V>::rank;
};

template <int R>
struct RankToType
{
};

template <>
struct RankToType<0>
{
    using type = int8_t;
};

template <>
struct RankToType<1>
{
    using type = int16_t;
};

template <>
struct RankToType<2>
{
    using type = int32_t;
};

template <>
struct RankToType<3>
{
    using type = int64_t;
};

/*
template <>
struct RankToType<4>
{
    using type = int;
};
*/

template <>
struct RankToType<4>
{
    using type = float;
};

template <>
struct RankToType<5>
{
    using type = double;
};




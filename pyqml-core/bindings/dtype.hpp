#pragma once
#include <variant>
#include <tuple>
#include "../cpp/src/tensor.hpp"

enum class DType
{
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

struct Tensor
{
    std::variant<tensor<int64_t>, tensor<int32_t>, tensor<int16_t>, tensor<int8_t>, tensor<float>, tensor<double>> tens;
    DType dtype;
    template <typename T>
    Tensor(tensor<T> &&tqn, DType typ) : tens(tqn), dtype(typ) {}
    std::vector<size_t> shape()
    {
        return std::visit([](const auto &t)
                          { return t.shape(); }, tens);
    }

    /*
    size_t ndim()
    {
        return std::visit([](const auto &t)
                          { return t.ndim(); }, tens);
    }
                          */

    Tensor operator+(const Tensor &other)
    {
        return std::visit([&](auto &&t1, auto &&t2) -> Tensor
                          {
        using T1 = std::decay_t<decltype(t1)>;
        using T2 = std::decay_t<decltype(t2)>;

        using U = T1::Data_type;
        using V = T2::Data_type;

        using R = RankToType<promote<U, V>::rank>::type;

        auto add = [](R a, R b) { return a + b; };

        auto result = binary_ops(t1, t2, add);

        DType res_type = (static_cast<int>(dtype) > static_cast<int>(other.dtype)) ? dtype : other.dtype;

        return Tensor(std::move(result), res_type); }, tens, other.tens);
    }
    Tensor operator-(const Tensor &other)
    {
        return std::visit([&](auto &&t1, auto &&t2) -> Tensor
                          {
        using T1 = std::decay_t<decltype(t1)>;
        using T2 = std::decay_t<decltype(t2)>;

        using U = T1::Data_type;
        using V = T2::Data_type;

        using R = RankToType<promote<U, V>::rank>::type;

        auto add = [](R a, R b) { return a - b; };

        auto result = binary_ops(t1, t2, add);

        DType res_type = (static_cast<int>(dtype) > static_cast<int>(other.dtype)) ? dtype : other.dtype;

        return Tensor(std::move(result), res_type); }, tens, other.tens);
    }
    Tensor argmax(int a)
    {
    }
    std::string print_val(std::stringstream &out)
    {
        return std::visit([&](auto &&t1)
                          { return get_str(out, t1); }, tens);
    }
};

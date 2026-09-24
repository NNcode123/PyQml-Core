#include "Tensor.hpp"
#include "dispatch.hpp"


template <typename Prop>
auto Tensor::getProp(Prop &&prop) const
{
    return Tensor::dispatch(dtype, [&](auto val)
                             {
                                 using T = std::decay_t<decltype(val)>;
                                 tensor<T> tens = tensor<T>::tensor_view(data, shape_, strides_, offset, size);
                                 return prop(tens);
                             });
}


template <typename Prop>
Tensor Tensor::getTens(Prop &&prop) const
{
    GET_TENSOR_PROP((*this), std::forward<Prop>(prop))
}


template <typename R>
Tensor Tensor::operator+(R value) const
{
    return getTens([&](auto &t)
                   { return t.operator+(value); });
}


template <typename R>
Tensor Tensor::operator-(R value) const
{
    return getTens([&](auto &t)
                   { return t.operator-(value); });
}


template <typename R>
Tensor Tensor::operator*(R value) const
{
    return getTens([&](auto &t)
                   { return t.operator*(value); });
}


template <typename R>
Tensor Tensor::operator/(R value) const
{
    return getTens([&](auto &t)
                   { return t.operator/(value); });
}


template <typename R>
Tensor operator+(R value, const Tensor &t)
{
    return t.getTens([&](auto &inner)
                     { return value + inner; });
}


template <typename R>
Tensor operator-(R value, const Tensor &t)
{
    return t.getTens([&](auto &inner)
                     { return value - inner; });
}


template <typename R>
Tensor operator*(R value, const Tensor &t)
{
    return t.getTens([&](auto &inner)
                     { return value * inner; });
}


template <typename R>
Tensor operator/(R value, const Tensor &t)
{
    return t.getTens([&](auto &inner)
                     { return value / inner; });
}


template <typename T>
Tensor Tensor::fill(const std::vector<size_t> &shape, T value, DType type)
{
    PYQ_UNARY_DISPATCH(type,
                       auto t_value = static_cast<atype>(value);
                       auto t_tensor = typed_fill(shape, t_value);
                       return Tensor(t_tensor.owner(), shape, type);)
}


template <typename T>
Tensor Tensor::arange(T start, T end, T step, DType dtype)
{
    return Tensor::dispatch(dtype, [&](auto typing)
                             {
                                 using R = std::decay_t<decltype(typing)>;
                                 size_t size = static_cast<size_t>(std::ceil((end - start) / step));
                                 if ((start >= end && step > 0) || (start <= end && step < 0))
                                 {
                                     size = 0;
                                 }
                                 R current = static_cast<R>(start);
                                 R increment = static_cast<R>(step);
                                 StorageRef out(new R[size], size);
                                 R *raw = out.data_ptr<R>();
                                 for (size_t index = 0; index < size; ++index)
                                 {
                                     *raw++ = current;
                                     current += increment;
                                 }
                                 return Tensor(out, {size}, dtype); });
}


template <typename... Slices>
Tensor Tensor::slice(const Slices &...slice_obj) const
{
    return getTens([&](auto &tens)
                   { return tens.slice(slice_obj...); });
}


template <typename... Slice>
Tensor Tensor::slice_view(const Slice &...slice_obj) const
{
    return getTens([&](auto &tens)
                   { return tens.slice_view(slice_obj...); });
}

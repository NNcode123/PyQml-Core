
#include "Tensor.hpp"
#include "dispatch.hpp"





/*
template <typename Prop>
Tensor getTens(Prop &&prop)
{
    PYQ_TENSOR_BUILD((*this), dtype,

                     auto tens = Tensor::getTens<type>(*this);
                     auto res = prop(tens);

                     return Tensor(
                         res.owner(),
                         res.dim(),
                         res.strides(),
                         dtype,
                         res.ofst());)
}

template <typename T>
static Tensor fill(const std::vector<size_t> &shape,
                   T value,
                   DType dtype)
{
    PYQ_TENSOR_BUILD(dummy, dtype,

                     size_t n_size = 1;
                     for (auto s : shape)
                         n_size *= s;

                     std::shared_ptr<type[]> data_n(new type[n_size]);

                     std::fill(
                         data_n.get(),
                         data_n.get() + n_size,
                         static_cast<type>(value));

                     return Tensor(data_n, shape, dtype);)
}

template <typename T>
static Tensor arange(T start, T end, T step, DType dtype)
{
    PYQ_TENSOR_BUILD(dummy, dtype,

                     size_t size = static_cast<size_t>(std::ceil((end - start) / step));

                     if ((start >= end && step > 0) || (start <= end && step < 0)) size = 0;

                     type cur = static_cast<type>(start); type stp = static_cast<type>(step);

                     std::shared_ptr<type[]> out(new type[size]);

                     for (size_t i = 0; i < size; ++i) {
                             out[i] = cur;
                             cur += stp; }

                     return Tensor(out, {size}, dtype);)
}
 */


template <typename Prop>
auto Tensor::getProp(Prop &&prop)
    {

        return Tensor::dispatch(dtype, [&](auto val)
                                {
            using T =std::decay_t<decltype(val)>;
            T* raw = static_cast<T*>(data.get());
            std::shared_ptr<T[]> data_n(data,raw);
            tensor<T> tens = tensor<T>::tensor_view(data_n,shape_, strides_, offset, size);
            return prop(tens); });
    }
    // This helper mirrors the data into the core tensor representation, applies a transform
    // to produce a new tensor, and then re-wraps the result as a Python-facing Tensor object.
template <typename Prop>
Tensor Tensor::getTens(Prop &&prop)
    {
        /*
        return Tensor::dispatch(dtype, [&](auto val)
                                {
            using T =std::decay_t<decltype(val)>;
            T* raw = static_cast<T*>(data.get());
            std::shared_ptr<T[]> data_n(data,raw);
            tensor<T> tens = tensor<T>::tensor_view(data_n, shape_, strides_, offset, size);
            tensor<T> res_tens= prop(tens);
        return  Tensor(res_tens.owner(),res_tens.dim(),res_tens.strides(), dtype, res_tens.ofst()); });
        */
       GET_TENSOR_PROP((*this),std::forward<Prop>(prop) )
    }
    // This convenience method converts the tensor contents into a printable string so the
    // Python binding can expose a readable representation through __repr__.
    std::string Tensor::print_val()
    {
        return getProp([](auto &t)
                       { return get_str(t); });
    }

    // This wrapper reduces the tensor along a chosen axis and returns the maximum values as
    // a new Tensor with the reduced dimension removed.
    Tensor Tensor::max(int axis)
    {
        return getTens([&](auto &t)
                       { return t.max(axis); });
    }

    // This wrapper reduces the tensor along a chosen axis and returns the minimum values as
    // a new Tensor, which is useful for summarizing data in a shape-preserving way.
    Tensor Tensor::min(int axis)
    {
        return getTens([&](auto &t)
                       { return t.min(axis); });
    }

    // template <typename Slice ...>

    // This factory creates a tensor filled with a single scalar value and a requested dtype,
    // which is useful for initialization patterns such as ones, zeroes, and constant masks.
    template <typename T>
     Tensor Tensor::fill(const std::vector<size_t> &shape, T value, DType type)
    {

        return Tensor::dispatch(type, [&](auto init_type)
                                {
            using R = std::decay_t<decltype(init_type)>;
            size_t n_size = 1;
            for (const auto& val: shape) {n_size *= val;}
            std::shared_ptr<R[]> data_n(new R[n_size]);
            std::fill(data_n.get(),data_n.get()+n_size,static_cast<R>(value));
            return Tensor(data_n, shape, type); });
    }

    // This factory constructs a tensor of ones so callers can initialize arrays with a
    // simple, explicit baseline value for numerical experiments and tests.
     Tensor Tensor::ones(const std::vector<size_t> &shape, DType type)
    {
        return Tensor::fill(shape, 1, type);
    }

    // This factory constructs a tensor of zeroes so callers can initialize arrays with a
    // neutral value before populating them through later operations.
     Tensor Tensor::zeroes(const std::vector<size_t> &shape, DType type)
    {
        return Tensor::fill(shape, 0, type);
    }

    // This factory builds a linearly spaced tensor from a start, end, and step value,
    // mirroring NumPy-style range generation for simple numerical utilities.
    template <typename T>
     Tensor Tensor::arange(T start, T end, T step, DType dtype)
    {

        return Tensor::dispatch(dtype, [&](auto typing)
                                {
                                    using R = std::decay_t<decltype(typing)>;
                                    size_t size = static_cast<size_t>(std::ceil((end - start) / (step)));
                                    if ((start >= end && step > 0) || (start <= end && step < 0)) size = 0;
                                    R strt = static_cast<R>(start);
                                    R stp = static_cast<R>(step);
                                    std::shared_ptr<R[]> out(new R[size]);
                                    R *raw = out.get();
                                    for (size_t j = 0; j < size; ++j)
                                    {
                                        *raw++ = strt;
                                        strt += stp;
                                    }
                                    return Tensor(out, {size}, dtype); });
    }

    // This method reshapes the logical tensor layout without changing the underlying data,
    // which is useful when a caller wants to reinterpret a flat buffer with a different shape.
    Tensor Tensor::reshape(const std::vector<size_t> &shape)
    {

        return getTens([&](auto &t)
                       { return t.reshape(shape); });
    }




    template <typename... Slices>

    Tensor Tensor::slice(const Slices &...slice_obj)
    {
        return getTens([&](auto &tens)
                       { return tens.slice(slice_obj...); });
    }

    // This template method produces a lightweight view of the tensor for slicing operations
    // that should reuse the underlying storage rather than allocate a fresh copy.
    template <typename... Slice>
    Tensor Tensor::slice_view(const Slice &...slice_obj)
    {
        return getTens([&](auto &tens)
                       { return tens.slice_view(slice_obj...); });
    }

    // This method exports the tensor into a NumPy-compatible pybind11 array so Python code
    // can inspect or further process the native data without extra conversion helpers.

    
   
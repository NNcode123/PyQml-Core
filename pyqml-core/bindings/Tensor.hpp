#pragma once
#include "dtype.hpp"
#include "bindings.hpp"

class Tensor
{

    std::shared_ptr<void> data;
    std::vector<size_t> shape_;
    std::vector<int64_t> strides_;
    size_t size;
    size_t offset;
    DType dtype;

public:
    template <typename F>
    static auto dispatch(DType dt, F &&f)
    {
        switch (dt)
        {
        case DType::Int8:
            return f(int8_t{});
        case DType::Int16:
            return f(int16_t{});
        case DType::Int32:
            return f(int32_t{});
        case DType::Int64:
            return f(int64_t{});
        case DType::Float32:
            return f(float{});
        case DType::Float64:
            return f(double{});
        }
        throw std::runtime_error("Unsupported DType");
    }

    template <typename F>
    static auto disp_2(DType a, DType b, F &&f)
    {
        return dispatch(a, [&](auto ta)
                        { return dispatch(b, [&](auto tb)
                                          { return f(ta, tb); }); });
    }

    template <typename T>
    Tensor(const std::shared_ptr<T[]> &owner, std::vector<size_t> dim, DType type) : shape_(dim),
                                                                                     offset(0), dtype(type)
    {
        int64_t stride = 1;
        strides_.resize(dim.size());
        for (int j = dim.size() - 1; j >= 0; --j)
        {
            strides_[j] = stride;
            stride *= dim[j];
        }
        size_t v = 1;
        for (const auto &val : dim)
        {
            v *= val;
        }
        size = v;
        data = std::static_pointer_cast<void>(owner);
        // data  = std::static_pointer_cast<void>(cuda_alloc<T>(size));
    }

    template <typename T>
    Tensor(const std::shared_ptr<T[]> &owner, const std::vector<size_t> &dim, const std::vector<int64_t> &stride, DType type, size_t off = 0) : shape_(dim), offset(off), strides_(stride), dtype(type)
    {

        size_t v = 1;
        for (const auto &val : dim)
        {
            v *= val;
        }
        size = v;
        data = std::static_pointer_cast<void>(owner);
       
    }

    template <typename T>
    Tensor(const py::array_t<T, py::array::c_style | py::array::forcecast> &array, const std::vector<size_t> &dim, DType type) : shape_(dim), offset(0), dtype(type)
    {
        auto arr_info = array.request();
        T *ptr = static_cast<T *>(arr_info.ptr);
        py::object owner = array;

        data = std::shared_ptr<T[]>(ptr, [owner](T *) mutable
                                    { owner = py::none(); });
        int64_t stride = 1;
        strides_.resize(dim.size());
        for (int j = dim.size() - 1; j >= 0; --j)
        {
            strides_[j] = stride;
            stride *= dim[j];
        }
        size_t v = 1;
        for (const auto &val : dim)
        {
            v *= val;
        }
        size = v;
    }

    template <typename T>
    Tensor(const std::vector<T> &val, const std::vector<size_t> &dim, DType type) : shape_(dim), offset(0), dtype(type)
    {
        auto ptr = std::shared_ptr<T[]>(new T[val.size()]);
        std::copy(val.begin(), val.end(), ptr.get());
        data = std::static_pointer_cast<void>(ptr);
        int64_t stride = 1;
        strides_.resize(dim.size());
        for (int j = dim.size() - 1; j >= 0; --j)
        {
            strides_[j] = stride;
            stride *= dim[j];
        }
        size_t v = 1;
        for (const auto &val : dim)
        {
            v *= val;
        }
        size = v;
    }

    template <typename Op>
    static Tensor dispatchOp(const Tensor &a, const Tensor &b, Op &&opy)
    {
        return Tensor::disp_2(a.dtype, b.dtype, [&](auto t1, auto t2)
                              {
        using T = std::decay_t<decltype(t1)>;
        using U = std::decay_t<decltype(t2)>;
        T* raw_a = static_cast<T*>(a.data.get());
        std::shared_ptr<T[]> data_a (a.data, raw_a);
        U* raw_b = static_cast<U*>(b.data.get());
        std::shared_ptr<U[]> data_b (b.data, raw_b);
        tensor<T> a_tens(data_a, a.shape_, a.strides_, a.offset, a.size);
        tensor<U> b_tens(data_b, b.shape_, b.strides_, b.offset, b.size);
        DType result = (static_cast<int>(a.dtype) > static_cast<int>(b.dtype)) ? a.dtype: b.dtype;
        auto tens = opy(a_tens, b_tens);
        return Tensor(tens.owner(), tens.dim(), result); });
    }

    template <typename Prop>
    auto getProp(Prop &&prop)
    {
        return Tensor::dispatch(dtype, [&](auto val)
                                {
            using T =std::decay_t<decltype(val)>;
            T* raw = static_cast<T*>(data.get());
            std::shared_ptr<T[]> data_n(data,raw);
            tensor<T> tens(data_n, shape_, strides_, offset, size);
            return prop(tens); });
    }
    template <typename Prop>
    Tensor getTens(Prop &&prop)
    {
        return Tensor::dispatch(dtype, [&](auto val)
                                {
            using T =std::decay_t<decltype(val)>;
            T* raw = static_cast<T*>(data.get());
            std::shared_ptr<T[]> data_n(data,raw);
            tensor<T> tens(data_n, shape_, strides_, offset, size);
            tensor<T> res_tens= prop(tens);
        return  Tensor(res_tens.owner(),res_tens.dim(),res_tens.strides(), dtype, res_tens.ofst()); });
    }
    std::string print_val()
    {
        return getProp([](auto &t)
                       { return get_str(t); });
    }

    Tensor max(int axis)
    {
        return getTens([&](auto &t)
                       { return t.max(axis); });
    }

    Tensor min(int axis)
    {
        return getTens([&](auto &t)
                       { return t.min(axis); });
    }

    // template <typename Slice ...>

    template <typename T>
    static Tensor fill(const std::vector<size_t> &shape, T value, DType type)
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

    static Tensor ones(const std::vector<size_t> &shape, DType type)
    {
        return Tensor::fill(shape, 1, type);
    }

    static Tensor zeroes(const std::vector<size_t> &shape, DType type)
    {
        return Tensor::fill(shape, 0, type);
    }

    template <typename T>
    static Tensor arange(T start, T end, T step, DType dtype)
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

    Tensor reshape(const std::vector<size_t> &shape)
    {

        return getTens([&](auto &t)
                       { return t.reshape(shape); });
    }

    Tensor operator+(const Tensor &other) const
    {
        return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::plus<>()); });
    }

    Tensor operator-(const Tensor &other) const
    {
        return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::minus<>()); });
    }

    Tensor operator*(const Tensor &other) const
    {
        return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::multiplies<>()); });
    }

    Tensor operator/(const Tensor &other) const
    {
        return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::divides<>()); });
    }

    Tensor astype(DType new_type, bool h) const
    {
        return Tensor::disp_2(dtype, new_type, [&](auto t1, auto t2)
                              {
        using T = std::decay_t<decltype(t1)>;
        using U = std::decay_t<decltype(t2)>;
        T* raw_a = static_cast<T*>(data.get());
        std::shared_ptr<T[]> data_a (data, raw_a);
        tensor<T> a_tens(data_a, shape_, strides_, offset, size);
        tensor<U> res = a_tens.template astype<U>(h);
        return Tensor(res.owner(), res.dim(), res.strides(), new_type, res.ofst()); });
    }

    template <typename... Slices>

    Tensor slice(const Slices &...slice_obj)
    {
        return getTens(dtype, [&](auto &tens)
                       { return tens.slice(slice_obj...); });
    }

    template <typename... Slice>
    Tensor slice_view(const Slice &...slice_obj)
    {
        return getTens(dtype, [&](auto &tens)
                       { return tens.slice_view(slice_obj...); });
    }

    py::array to_numpy()
    {
        return Tensor::dispatch(dtype, [&](auto val)
                                {
        using R = std::decay_t<decltype(val)>;

        
        std::shared_ptr<R[]> new_ptr(data, static_cast<R*>(data.get()));

        tensor<R> tens_view(new_ptr, shape_, strides_, offset, size);
        tensor<R> new_tens = tens_view.copy();

        std::vector<int64_t> numpy_strides = new_tens.strides();
        std::transform(
            numpy_strides.begin(),
            numpy_strides.end(),
            numpy_strides.begin(),
            [](auto s) { return s * sizeof(R); }
        );

        return py::array(
            py::buffer_info(
                new_tens.data(),                         // ptr
                sizeof(R),                               // itemsize
                py::format_descriptor<R>::format(),      // dtype
                new_tens.ndim(),                         // ndim
                new_tens.shape(),                        // shape
                numpy_strides                            // strides (bytes)
            )
        ); });
    }

    DType type() const
    {
        return dtype;
    }

    std::vector<size_t> shape() const { return shape_; }
};

#include "Tensor_ops/free_ops.cpp"
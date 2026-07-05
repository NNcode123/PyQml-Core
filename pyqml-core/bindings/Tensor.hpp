#pragma once
#include "dtype.hpp"
#include "bindings.hpp"
#include "../cpp/src/tensor.hpp"

#define DISPATCH_DTYPE(DTYPE, Func) \
    switch (DTYPE)                  \
    {                               \
    case DType::Int32:              \
        using out_t = int32_t;      \
        Func();                     \
    case DType::Int64:              \
        using out_t = int64_t;      \
        Func();                     \
    }

class Tensor
{

    std::shared_ptr<void> data;
    std::vector<size_t> shape_;
    std::vector<int64_t> strides_;
    size_t size;
    size_t offset;
    DType dtype;
    // std::shared_ptr<Node> grad_fn;

public:
    // This helper resolves a runtime DType into a concrete C++ scalar type and executes
    // the supplied callback with that type so the same tensor logic can be reused across dtypes.
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

    // This overload dispatches two runtime dtypes at once, which is useful when a tensor
    // operation needs to combine the storage types of both operands before producing a result.
    template <typename F>
    static auto disp_2(DType a, DType b, F &&f)
    {
        return dispatch(a, [&](auto ta)
                        { return dispatch(b, [&](auto tb)
                                          { return f(ta, tb); }); });
    }

    // This constructor wraps an existing buffer with a logical tensor shape and dtype so
    // Python-visible tensor objects can reference shared storage without copying the data.
    template <typename T>
    Tensor(const std::shared_ptr<T[]> &owner, std::vector<size_t> dim, DType type) : shape_(dim),
                                                                                     offset(0), dtype(type), size(calc_size(dim))
    {

        fill_size_vec(dim, strides_);
        data = std::static_pointer_cast<void>(owner);
        // data  = std::static_pointer_cast<void>(cuda_alloc<T>(size));
    }

    // This constructor creates a view over an existing buffer with explicit strides and an
    // optional offset, which lets the wrapper represent slices and broadcasted views efficiently.
    template <typename T>
    Tensor(const std::shared_ptr<T[]> &owner, const std::vector<size_t> &dim, const std::vector<int64_t> &stride, DType type, size_t off = 0) : shape_(dim), offset(off), strides_(stride), dtype(type),
                                                                                                                                                size(calc_size(dim))
    {
        data = std::static_pointer_cast<void>(owner);
    }

    // This constructor imports a pybind11 NumPy array into a Tensor wrapper while preserving
    // the requested logical shape and attached dtype metadata for downstream operations.

    template <typename T>
    Tensor(const py::array_t<T, py::array::c_style | py::array::forcecast> &array, const std::vector<size_t> &dim, DType type) : shape_(dim), offset(0), dtype(type), size(calc_size(dim))
    {
        auto arr_info = array.request();
        T *ptr = static_cast<T *>(arr_info.ptr);
        py::object owner = array;

        data = std::shared_ptr<T[]>(ptr, [owner](T *) mutable
                                    { owner = py::none(); });
        fill_size_vec(dim, strides_);
    }

    // This constructor materializes a tensor from a standard vector and shape description,
    // making it straightforward to build native tensors from Python lists or other host data.
    template <typename T>
    Tensor(const std::vector<T> &val, const std::vector<size_t> &dim, DType type) : shape_(dim), offset(0), dtype(type), size(calc_size(dim))
    {
        auto ptr = std::shared_ptr<T[]>(new T[val.size()]);
        std::copy(val.begin(), val.end(), ptr.get());
        data = std::static_pointer_cast<void>(ptr);
        fill_size_vec(dim, strides_);
    }

    // This helper dispatches elementwise operations by matching the runtime dtypes of both
    // inputs and then delegating to the core tensor implementation with the appropriate scalar types.
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
        tensor<T> a_tens = tensor<T>::tensor_view(data_a, a.shape_, a.strides_, a.offset, a.size);
        tensor<U> b_tens = tensor<U>::tensor_view(data_b, b.shape_, b.strides_, b.offset, b.size);
        DType result = (static_cast<int>(a.dtype) > static_cast<int>(b.dtype)) ? a.dtype: b.dtype;
        auto tens = opy(a_tens, b_tens);
        return Tensor(tens.owner(), tens.dim(), result); });
    }

    // This helper provides a bridge from the wrapper tensor to the core tensor engine by
    // constructing a concrete tensor view with the current dtype and running a probe operation on it.
    template <typename Prop>
    auto getProp(Prop &&prop)
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
    Tensor getTens(Prop &&prop)
    {
        return Tensor::dispatch(dtype, [&](auto val)
                                {
            using T =std::decay_t<decltype(val)>;
            T* raw = static_cast<T*>(data.get());
            std::shared_ptr<T[]> data_n(data,raw);
            tensor<T> tens = tensor<T>::tensor_view(data_n, shape_, strides_, offset, size);
            tensor<T> res_tens= prop(tens);
        return  Tensor(res_tens.owner(),res_tens.dim(),res_tens.strides(), dtype, res_tens.ofst()); });
    }
    // This convenience method converts the tensor contents into a printable string so the
    // Python binding can expose a readable representation through __repr__.
    std::string print_val()
    {
        return getProp([](auto &t)
                       { return get_str(t); });
    }

    // This wrapper reduces the tensor along a chosen axis and returns the maximum values as
    // a new Tensor with the reduced dimension removed.
    Tensor max(int axis)
    {
        return getTens([&](auto &t)
                       { return t.max(axis); });
    }

    // This wrapper reduces the tensor along a chosen axis and returns the minimum values as
    // a new Tensor, which is useful for summarizing data in a shape-preserving way.
    Tensor min(int axis)
    {
        return getTens([&](auto &t)
                       { return t.min(axis); });
    }

    // template <typename Slice ...>

    // This factory creates a tensor filled with a single scalar value and a requested dtype,
    // which is useful for initialization patterns such as ones, zeroes, and constant masks.
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

    // This factory constructs a tensor of ones so callers can initialize arrays with a
    // simple, explicit baseline value for numerical experiments and tests.
    static Tensor ones(const std::vector<size_t> &shape, DType type)
    {
        return Tensor::fill(shape, 1, type);
    }

    // This factory constructs a tensor of zeroes so callers can initialize arrays with a
    // neutral value before populating them through later operations.
    static Tensor zeroes(const std::vector<size_t> &shape, DType type)
    {
        return Tensor::fill(shape, 0, type);
    }

    // This factory builds a linearly spaced tensor from a start, end, and step value,
    // mirroring NumPy-style range generation for simple numerical utilities.
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

    // This method reshapes the logical tensor layout without changing the underlying data,
    // which is useful when a caller wants to reinterpret a flat buffer with a different shape.
    Tensor reshape(const std::vector<size_t> &shape)
    {

        return getTens([&](auto &t)
                       { return t.reshape(shape); });
    }

    // This overload implements elementwise addition between two tensors and returns a new
    // tensor that preserves the broadcasted shape of the operands.
    Tensor operator+(const Tensor &other) const
    {
        return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::plus<>()); });
    }

    // This overload implements elementwise subtraction between two tensors and returns the
    // result as a new tensor with the broadcasted shape of the inputs.
    Tensor operator-(const Tensor &other) const
    {
        return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::minus<>()); });
    }

    // This overload implements elementwise multiplication between two tensors and returns a
    // new tensor that reflects the broadcasted shape of the operands.
    Tensor operator*(const Tensor &other) const
    {
        return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::multiplies<>()); });
    }

    // This overload implements elementwise division between two tensors and returns the
    // quotient as a new tensor while preserving the broadcasted shape semantics.
    Tensor operator/(const Tensor &other) const
    {
        return dispatchOp(*this, other, [&](auto &t_1, auto &t_2)
                          { return binary_ops(t_1, t_2, std::divides<>()); });
    }

    // This method converts the tensor to a different dtype and optionally copies the memory,
    // enabling explicit type control when interacting with Python or native code.
    Tensor astype(DType new_type, bool h) const
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

    // This template method extracts a view or copy of the tensor using one or more slice
    // specifications, making the wrapper support the same selection patterns as NumPy.
    template <typename... Slices>

    Tensor slice(const Slices &...slice_obj)
    {
        return getTens(dtype, [&](auto &tens)
                       { return tens.slice(slice_obj...); });
    }

    // This template method produces a lightweight view of the tensor for slicing operations
    // that should reuse the underlying storage rather than allocate a fresh copy.
    template <typename... Slice>
    Tensor slice_view(const Slice &...slice_obj)
    {
        return getTens(dtype, [&](auto &tens)
                       { return tens.slice_view(slice_obj...); });
    }

    // This method exports the tensor into a NumPy-compatible pybind11 array so Python code
    // can inspect or further process the native data without extra conversion helpers.

    py::array to_numpy()
    {
        return Tensor::dispatch(dtype, [&](auto val)
                                {
        using R = std::decay_t<decltype(val)>;


        std::vector<int64_t> numpy_strides = strides_;
        void * DATA = static_cast<R*>(data.get()) + offset;
        std::transform(
            numpy_strides.begin(),
            numpy_strides.end(),
            numpy_strides.begin(),
            [](auto s) { return s * sizeof(R); }
        );

        std::vector<py::ssize_t> n_shape;
        n_shape.reserve(shape_.size());
        for (auto val: shape_)
            n_shape.push_back(static_cast<py::ssize_t>(val));

        return py::array(
           py::memoryview::from_buffer(
                DATA,                        // ptr
                sizeof(R),                               // itemsize
                py::format_descriptor<R>::value,      // dtype                       // ndim
                n_shape,                        // shape
                numpy_strides                            // strides (bytes)
            )
        ); });
    }

    // This accessor returns the logical dtype of the tensor so callers can inspect how the
    // data is represented before performing additional operations.
    DType type() const
    {
        return dtype;
    }

    // This accessor returns the tensor shape so Python and C++ callers can inspect the
    // logical dimensions that describe the data layout.
    std::vector<size_t> shape() const { return shape_; }
};

#include "Tensor_ops/free_ops.cpp"
#include "dispatch.cpp"

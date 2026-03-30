#pragma once
#include "dtype.hpp"
#include "bindings.hpp"

template <typename... Ts>

DType inf_dtyp_python(py::dtype dt)
{
    DType result;
    bool found = false;
    ([&]()
     { if (!found && dt.is(py::dtype::of<Ts>())){
            result = typeDType<Ts>::type;
            found = true;
        } }(), ...);

    return result;
}

inline DType infer_types(py::object obj, bool obj_instance)
{
    if (obj_instance)
    {
        if (py::isinstance<py::int_>(obj))
        {
            return DType::Int64;
        }
        if (py::isinstance<py::float_>(obj))
        {
            return DType::Float64;
        }
    }
    else
    {
        if (py::isinstance<py::type>(obj))
        {
            py::object bins = py::module_::import("builtins");
            py::type Ty = py::cast<py::type>(obj);
            if (obj.is(bins.attr("int")))
            {
                return DType::Int64;
            }
            if (obj.is(bins.attr("float")))
            {
                return DType::Float64;
            }
        }
    }
    try
    {
        return py::cast<DType>(obj);
    }
    catch (const py::cast_error &e)
    {
        throw std::runtime_error("Unsupported object type");
    }
    throw std::runtime_error("Unsupported object type");
}

class Tensor
{
    // std::variant<tensor<int64_t>, tensor<int32_t>, tensor<int16_t>, tensor<int8_t>, tensor<float>, tensor<double>> tens;

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

        data = std::static_pointer_cast<void>(owner);
        strides_.resize(dim.size());

        int64_t stride = 1;
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
    Tensor(const std::shared_ptr<T[]> &owner, const std::vector<size_t> &dim, const std::vector<int64_t> &stride, DType type, size_t off = 0) : shape_(dim), offset(off), strides_(stride), dtype(type)
    {
        data = std::static_pointer_cast<void>(owner);
        size_t v = 1;
        for (const auto &val : dim)
        {
            v *= val;
        }
        size = v;
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
    Tensor dispatchOp(const Tensor &a, const Tensor &b, Op &&opy) const
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
        auto tens = binary_ops(a_tens,b_tens, opy);
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
                                    R __restrict*raw = out.get();
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
        return dispatchOp(*this, other, std::plus<>());
    }

    Tensor operator-(const Tensor &other) const
    {
        return dispatchOp(*this, other, std::minus<>());
    }

    Tensor operator*(const Tensor &other) const
    {
        return dispatchOp(*this, other, std::multiplies<>());
    }

    Tensor operator/(const Tensor &other) const
    {
        return dispatchOp(*this, other, std::divides<>());
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
                       { 
                        
                        return tens.slice_view(slice_obj...); });
    }

    DType type() const
    {
        return dtype;
    }

    std::vector<size_t> shape() const { return shape_; }

    /*

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

    /*
    const Tensor &operator+(const Tensor &other) const
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
    const Tensor &operator-(const Tensor &other) const
    {
        return std::visit([&](auto &&t1, auto &&t2) -> Tensor
                          {
        using T1 = std::decay_t<decltype(t1)>;
        using T2 = std::decay_t<decltype(t2)>;

        using U = T1::Data_type;
        using V = T2::Data_type;
a
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
    */
};

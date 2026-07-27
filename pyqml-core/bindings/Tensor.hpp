#pragma once
#include "dtype.hpp"
#include "Autograd/Node.hpp"
#include "../cpp/src/tensor.hpp"

struct grad_meta;


class Tensor
{

    pyq_intrusive_ptr<Storage> data;
    
    std::vector<size_t> shape_;
    std::vector<int64_t> strides_;
    size_t size;
    size_t offset;
    DType dtype;
    //pyq_intrusive_ptr<Node> grad_fn;
    //grad_meta info;

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

    template <typename T>
    tensor<T> get_typed_tensor() const
    {
        //pyq_intrusive_ptr<Storage> data_n  = data;
        tensor<T> tens = tensor<T>::tensor_view(data /*data*/, shape_, strides_, offset, size);
        return tens;
    }

    // This constructor wraps an existing buffer with a logical tensor shape and dtype so
    // Python-visible tensor objects can reference shared storage without copying the data.
    Tensor(const pyq_intrusive_ptr<Storage>&   owner, const std::vector<size_t>& dim, DType type) : shape_(dim),
                                                                                     offset(0), dtype(type), size(calc_size(dim))
    {

        fill_size_vec(dim, strides_);
        data = owner;
      
    }

    // This constructor creates a view over an existing buffer with explicit strides and an
    // optional offset, which lets the wrapper represent slices and broadcasted views efficiently.
    Tensor(const pyq_intrusive_ptr<Storage>& owner, const std::vector<size_t> &dim, const std::vector<int64_t> &stride, DType type, size_t off = 0) : shape_(dim), offset(off), strides_(stride), dtype(type),
                                                                                                                                                size(calc_size(dim))
    {
        data = owner;
    }

    // This constructor imports a pybind11 NumPy array into a Tensor wrapper while preserving
    // the requested logical shape and attached dtype metadata for downstream operations.

    
    

    // This constructor materializes a tensor from a standard vector and shape description,
    // making it straightforward to build native tensors from Python lists or other host data.
    template <typename T>
    Tensor(const std::vector<T> &val, const std::vector<size_t> &dim, DType type) : shape_(dim), offset(0), dtype(type), size(calc_size(dim))
    {
        data = make_intrusive<Storage,T>(new T[val.size()], val.size());
        std::copy(val.begin(), val.end(), data.template get<T>());
        fill_size_vec(dim, strides_);
    }

    // This helper dispatches elementwise operations by matching the runtime dtypes of both
    // inputs and then delegating to the core tensor implementation with the appropriate scalar types.

    // This helper provides a bridge from the wrapper tensor to the core tensor engine by
    // constructing a concrete tensor view with the current dtype and running a probe operation on it.
    template <typename Prop>
    auto getProp(Prop &&prop);
    // This helper mirrors the data into the core tensor representation, applies a transform
    // to produce a new tensor, and then re-wraps the result as a Python-facing Tensor object.
    template <typename Prop>
    Tensor getTens(Prop &&prop);
    // This convenience method converts the tensor contents into a printable string so the
    // Python binding can expose a readable representation through __repr__.
    std::string print_val();

    // This wrapper reduces the tensor along a chosen axis and returns the maximum values as
    // a new Tensor with the reduced dimension removed.
    Tensor max(int axis);

    // This wrapper reduces the tensor along a chosen axis and returns the minimum values as
    // a new Tensor, which is useful for summarizing data in a shape-preserving way.
    Tensor min(int axis);
    // template <typename Slice ...>

    // This factory creates a tensor filled with a single scalar value and a requested dtype,
    // which is useful for initialization patterns such as ones, zeroes, and constant masks.
    template <typename T>
    static Tensor fill(const std::vector<size_t> &shape, T value, DType type);

    // This factory constructs a tensor of ones so callers can initialize arrays with a
    // simple, explicit baseline value for numerical experiments and tests.
    static Tensor ones(const std::vector<size_t> &shape, DType type);

    // This factory constructs a tensor of zeroes so callers can initialize arrays with a
    // neutral value before populating them through later operations.
    static Tensor zeroes(const std::vector<size_t> &shape, DType type);

    // This factory builds a linearly spaced tensor from a start, end, and step value,
    // mirroring NumPy-style range generation for simple numerical utilities.
    template <typename T>
    static Tensor arange(T start, T end, T step, DType dtype);
    // This method reshapes the logical tensor layout without changing the underlying data,
    // which is useful when a caller wants to reinterpret a flat buffer with a different shape.
    Tensor reshape(const std::vector<size_t> &shape);

    // This overload implements elementwise addition between two tensors and returns a new
    // tensor that preserves the broadcasted shape of the operands.
    Tensor operator+(const Tensor &other) const;

    // This overload implements elementwise subtraction between two tensors and returns the
    // result as a new tensor with the broadcasted shape of the inputs.
    Tensor operator-(const Tensor &other) const;
    // This overload implements elementwise multiplication between two tensors and returns a
    // new tensor that reflects the broadcasted shape of the operands.
    Tensor operator*(const Tensor &other) const;
    // This overload implements elementwise division between two tensors and returns the
    // quotient as a new tensor while preserving the broadcasted shape semantics.
    Tensor operator/(const Tensor &other) const;

    // This method converts the tensor to a different dtype and optionally copies the memory,
    // enabling explicit type control when interacting with Python or native code.
    Tensor astype(DType new_type, bool h) const;

    // This template method extracts a view or copy of the tensor using one or more slice
    // specifications, making the wrapper support the same selection patterns as NumPy.
    template <typename... Slices>

    Tensor slice(const Slices &...slice_obj);

    // This template method produces a lightweight view of the tensor for slicing operations
    // that should reuse the underlying storage rather than allocate a fresh copy.
    template <typename... Slice>
    Tensor slice_view(const Slice &...slice_obj);

    // This method exports the tensor into a NumPy-compatible pybind11 array so Python code
    // can inspect or further process the native data without extra conversion helpers.


    // This accessor returns the logical dtype of the tensor so callers can inspect how the
    // data is represented before performing additional operations.
    DType type() const
    {
        return dtype;
    }

    // This accessor returns the tensor shape so Python and C++ callers can inspect the
    // logical dimensions that describe the data layout.
    std::vector<size_t> shape() const { return shape_; }

    std::vector<int64_t> strides() const {return strides_;}

    size_t get_size() const {return size;}

    size_t get_offset() const{return offset;}


    pyq_intrusive_ptr<Storage> data_ptr() const {return data;}


    
};


#include "dispatch_binary.hpp"
#include "dispatch_unary.hpp"

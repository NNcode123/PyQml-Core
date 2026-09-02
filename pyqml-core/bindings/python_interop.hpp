#include "Tensor.hpp"
#include "bindings.hpp"
#include "dispatch.hpp"



template <typename T>
Tensor tensor_from_python(const py::array_t<T, py::array::c_style | py::array::forcecast> &array, const std::vector<size_t> &dim, DType type)
{
        auto arr_info = array.request();
        
        T *ptr = static_cast<T *>(arr_info.ptr);
        
        py::object owner = array;

        auto data = std::shared_ptr<T[]>(ptr, [owner](T *) mutable
                                    { owner = py::none(); });

        return Tensor(data, dim, type );
}
        

 py::array to_numpy(const Tensor& tens)
    {
        return Tensor::dispatch(tens.type(), [&](auto val)
                                {
        using R = std::decay_t<decltype(val)>;


        std::vector<int64_t> numpy_strides = tens.strides();
        void * DATA = tens.void_data();
        std::transform(
            numpy_strides.begin(),
            numpy_strides.end(),
            numpy_strides.begin(),
            [](auto s) { return s * sizeof(R); }
        );

        auto shape = tens.shape();

        std::vector<py::ssize_t> n_shape;
        n_shape.reserve(shape.size());
        for (auto val: shape)
            n_shape.push_back(static_cast<py::ssize_t>(val));

        auto owner = new StorageRef(data);

        py::capsule base(owner, [](void *p) {
    delete static_cast<StorageRef *>(p);
    });


     

        return py::array(
                py::dtype::of<R>(),
                n_shape,                    // ptr
                numpy_strides,                              // itemsize
                DATA,    // dtype                       // ndim
                base
                                           // strides (bytes)
            
        ); });
    }
        




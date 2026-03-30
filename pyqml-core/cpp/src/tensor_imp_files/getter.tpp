#include "../tensor.hpp"

template <typename T>
void printTens(std::ostream &out, const tensor<T> &tensor, int64_t start, size_t depth = 1)
{

    std::vector<int64_t> stride = tensor.strides();
    std::vector<size_t> dim = tensor.dim();
    const T *__restrict data = tensor.data();

    int width = 0;
    if (width == 0)
    {
        /*
        for (const auto &val : data)
        {
            width = std::max(width, (int)std::to_string(val).size());
        }
            */
        for (size_t ind = 0; ind < tensor.size(); ++ind)
        {
            width = std::max(width, (int)std::to_string(data[ind]).size());
        }
    }

    if (depth == dim.size())
    {
        int dir = 1;
        int64_t cur_stride = stride[depth - 1];
        int64_t last_dim = dim[depth - 1];
        out << "[";
        for (size_t times = 0; times < last_dim; times++)
        {

            out << std::setw(width) << data[start] << " ";
            start += cur_stride;
        }
        out << "]";
        return;
    }

    int64_t cur_stride = stride[depth - 1];
    size_t num_sub_tensors = dim[depth - 1];
    out << "[";
    for (int64_t i = 0; i < num_sub_tensors; i++)
    {
        if (i > 0)
        {
            for (size_t pad = 0; pad < depth; pad++)
            {
                out << " ";
            }
        }

        printTens(out, tensor, start + i * cur_stride, depth + 1);
        if (i != num_sub_tensors - 1)
            out << "\n\n";
    }
    out << "]";
}

template <typename T>
std::string get_str(const tensor<T> &tensor)
{
    std::stringstream out;
    printTens(out, tensor, 0);
    return out.str();
}
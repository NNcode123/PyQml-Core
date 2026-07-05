#include <vector>

void fill_size_vec(const std::vector<size_t> &dim, std::vector<int64_t> &strides)
{
    strides.resize(dim.size());
    size_t expect = 1;
    for (int i = dim.size() - 1; i >= 0; i--)
    {
        strides[i] = expect;
        expect *= dim[i];
    }
}

size_t calc_size(const std::vector<size_t> &dim)
{
    size_t u = 1;
    for (auto val : dim)
    {
        u *= val;
    }
    return u;
}
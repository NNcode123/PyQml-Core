#include <vector>
#include <complex.h>

using size_t = std::size_t;
template <typename T>
struct typing
{
    // To be Defined Later....
};

template <typename T>
class tensor
{
    std::vector<T> data_;
    std::vector<size_t> dim_;

public:
    explicit tensor(const std::vector<T> &data, const std::vector<size_t> &dim_ =);
    [[nodiscard]] size_t size() const;
    [[nodiscard]] T at(const std::vector<size_t> &pos) const;
    template <typename... Indices>
    [[nodiscrad]] T operator()(Indices... indices) const;
    [[nodiscard]] const std::vector<size_t> &shape() const;
    void reshape(const std::vector<size_t> &newshape);
    Tensor &matrixPow(const size_t &val);
    Tensor &elemPow(const size_t &val);
    friend : std::ostream &operator<<(std::ostream out, const Tensor &tensor) const;
};

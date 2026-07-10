#include "Tensor.hpp"
#include <numeric>
#include <vector>

int main()
{
    std::vector<int8_t> q;
    std::vector<float> u;
    q.resize(200);
    u.resize(200);
    std::iota(q.begin(), q.end(), 1);
    std::iota(u.begin(), u.end(), 1);

    Tensor tens(q, {10, 10, 2}, DType::Int8);
    Tensor tens_1(u, {10, 10, 2}, DType::Float32);
    auto expr = tens - tens_1+tens*tens_1/(tens_1*tens);
    std::cout << expr.print_val() << std::endl;
}
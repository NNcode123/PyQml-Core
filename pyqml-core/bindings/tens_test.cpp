#include "Tensor.hpp"
#include <numeric>
#include <vector>

int main()
{
    std::vector<int32_t> q;
    std::vector<float> u;
    q.resize(200);
    u.resize(200);
    std::iota(q.begin(), q.end(), 1);
    std::iota(u.begin(), u.end(), 1);

    Tensor tens(q, {10, 10, 2}, DType::Int32);
    Tensor tens_1(u, {10, 10, 2}, DType::Float32);
    auto expr = tens + tens_1;
    auto expr_1 =  expr.astype(DType::Int32, false);
    auto expr_2 = expr.slice(Range({5,3,8}), Range({8,2,0}));
    std::cout << expr_2.print_val() << "\n";
   // std::cout << "integer_part: " << expr_1.print_val() << "\n";
}
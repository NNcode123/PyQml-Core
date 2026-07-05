#include "Tensor.hpp"
#include <numeric>
#include <vector>


int main() {
    std::vector<int> q;
    q.resize(200);
    Tensor tens(q, {10,10,2}, DType::Int32);
    std::cout << tens.print_val() << std::endl;
}
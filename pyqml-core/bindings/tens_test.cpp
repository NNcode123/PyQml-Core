#include "Tensor.hpp"
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

void build_nested_views(const Tensor &base, std::vector<Tensor> *out)
{
    for (int pass = 0; pass < 8; ++pass)
    {
        {
            Tensor first = base.slice_view(
                Slice(0 + (pass % 2), 8 + (pass % 3), 1),
                Slice(1 + (pass % 2), 7 + (pass % 2), 2),
                Slice(0, 4, 1));

            {
                Tensor second = first.slice_view(
                    Slice(0, 3, 1),
                    Slice(0, 3, 1),
                    Slice(0, 2, 1));

                {
                    Tensor third = second.slice_view(
                        Slice(0, 2, 1),
                        Slice(1, 3, 1),
                        Slice(0, 1, 1));

                    {
                        Tensor fourth = third.slice_view(
                            Slice(0, 1, 1),
                            Slice(0, 1, 1),
                            Slice(0, 1, 1));

                        if (out != nullptr)
                        {
                            out->push_back(fourth);
                        }
                    }
                }
            }
        }
    }
}

} // namespace

int main()
{
    /*
    std::vector<int32_t> values(1024);
    std::iota(values.begin(), values.end(), 1);

    Tensor root(values, {16, 8, 8}, DType::Int32);

    std::vector<Tensor> holders;
    holders.reserve(160);

    build_nested_views(root, &holders);

    for (int i = 0; i < 32; ++i)
    {
        Tensor chain = root.slice_view(
            Slice(0, 8, 1),
            Slice(i % 4, 8, 1),
            Slice(0, 4, 1));

        Tensor nested = chain.slice_view(
            Slice(0, 2, 1),
            Slice(0, 2, 1),
            Slice(0, 2, 1));

        Tensor copy = nested;
        holders.push_back(copy);
    }

    for (int outer = 0; outer < 4; ++outer)
    {
        {
            Tensor lhs = root.slice_view(Slice(0, 4, 1), Slice(0, 4, 1), Slice(0, 4, 1));
            Tensor rhs = lhs.slice_view(Slice(0, 2, 1), Slice(0, 2, 1), Slice(0, 2, 1));
            Tensor deep = rhs.slice_view(Slice(0, 1, 1), Slice(0, 1, 1), Slice(0, 1, 1));

            holders.push_back(deep);
            {
                Tensor extra = deep;
                holders.push_back(extra);
            }
        }
    }

    holders.clear();

    {
        Tensor final_view = root.slice_view(Slice(2, 10, 2), Slice(1, 7, 1), Slice(0, 4, 1));
        Tensor final_copy = final_view;
        std::string printed = final_view.print_val();
        assert(!printed.empty());
    }

    std::cout << "pathological view suite completed" << std::endl;
    return 0;
    */

    std::vector<int64_t> data(100);

for (int64_t i = 0; i < 100; ++i)
    data[i] = i + 1;

Tensor t(data, {2, 5, 2, 5}, DType::Int64);

std::cout << "=============================\n";
std::cout << "Original Tensor\n";
std::cout << "=============================\n"
          << t.print_val() << "\n";

//------------------------------------
// Sum axis {1}
//------------------------------------

Tensor out1 = t.sum({1}, false);

std::cout << "\n=============================\n";
std::cout << "sum({1}, false)\n";
std::cout << "=============================\n"
          << out1.print_val() << "\n";

Tensor out2 = t.sum({1}, true);

std::cout << "\n=============================\n";
std::cout << "sum({1}, true)\n";
std::cout << "=============================\n"
          << out2.print_val() << "\n";

//------------------------------------
// Sum axes {1,3}
//------------------------------------

Tensor out3 = t.sum({1,3}, false);

std::cout << "\n=============================\n";
std::cout << "sum({1,3}, false)\n";
std::cout << "=============================\n"
          << out3.print_val() << "\n";

Tensor out4 = t.sum({1,3}, true);

std::cout << "\n=============================\n";
std::cout << "sum({1,3}, true)\n";
std::cout << "=============================\n"
          << out4.print_val() << "\n";

//------------------------------------
// Sum axes {0,2,3}
//------------------------------------

Tensor out5 = t.sum({0,2,3}, false);

std::cout << "\n=============================\n";
std::cout << "sum({0,2,3}, false)\n";
std::cout << "=============================\n"
          << out5.print_val() << "\n";

Tensor out6 = t.sum({0,2,3}, true);

std::cout << "\n=============================\n";
std::cout << "sum({0,2,3}, true)\n";
std::cout << "=============================\n"
          << out6.print_val() << "\n";

//------------------------------------
// Sum all axes
//------------------------------------


/*

Tensor out7 = t.sum({0,1,2,3}, false);

std::cout << "\n=============================\n";
std::cout << "sum({0,1,2,3}, false)\n";
std::cout << "=============================\n"
          << out7.print_val() << "\n";

Tensor out8 = t.sum({0,1,2,3}, true);

std::cout << "\n=============================\n";
std::cout << "sum({0,1,2,3}, true)\n";
std::cout << "=============================\n"
          << out8.print_val() << "\n";

          */

return 0;
}
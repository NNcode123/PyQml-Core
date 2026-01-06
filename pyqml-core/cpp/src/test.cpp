#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <complex.h>
#include <chrono>

template <typename F>
void time_block(const std::string &name, F &&fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << name << ": "
              << std::chrono::duration<double>(t1 - t0).count()
              << " sec\n";
}

int main()
{
    /*
    std::cout << 2 << std::endl;

    std::vector<int> dataN = {
        1, 2,
        3, 4,
        5, 6,
        7, 8};

    // Shape: 2 x 2 x 2
    std::vector<size_t> dims = {2, 2, 2};

    tensor<int> t = tensor<int>(dataN, dims);

    std::cout << std::endl;
    std::cout << t.slice(Slice(1, -3, -1), Slice(1, -3, -1), Slice(1, -3, -1));

    std::vector<int> datas(100);
    for (int i = 0; i < 100; ++i)
    {
        datas[i] = i;
    }

    // Shape: 2 x 3 x 5  → 2*3*5 = 30
    std::vector<size_t> dim = {5, 5, 2, 2};

    tensor<int> tens(datas, dim);

    std::cout << "Tensor shape: {5, 5, 2, 2}\n";
    std::cout << tens << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << tens.slice_view(Slice(-1, -4, -1), Slice(-1, -5, -2)) << std::endl;
    std::cout << "\n\n";
    tensor<int> slice_tens = tens.slice(Slice(-1, -4, -1), Slice(-1, -5, -2));
    std::cout << std::endl;
    std::cout << slice_tens << std::endl;

    /*
    std::vector<int> data_x(40);
    for (int i = 0; i < 40; ++i)
    {
        data_x[i] = i;
    }

    // Shape: 5 x 4 x 2  → 5*4*2 = 40

    std::vector<size_t> dimensions = {5, 4, 2};

    tensor<int> tensS(data_x, dimensions);

    std::cout << "Base tensor tensS (shape {5,4,2}):\n";
    std::cout << tensS << "\n\n";

    // ----------- First slice: VIEW (all negative) -----------
    auto v = tensS.slice_view(
        Slice(-1, -6, -1), // axis 0 → [4,3,2,1,0]
        Slice(-1, -5, -2), // axis 1 → [3,1]
        Slice(-1, -3, -1)  // axis 2 → [1,0]
    );

    std::cout << "Slice view v:\n";
    std::cout << v << "\n\n";

    // ----------- Second slice: COPY (still negative) -----------
    auto c = v.slice(
        Slice(-1, -4, -1), // reverse first 3 along axis 0
        Slice(-1, -3, -1), // reverse axis 1
        Slice(-1, -2, -1)  // take only first element of axis 2
    );

    std::cout << "Slice of slice_view (deep copy):\n";
    std::cout << c << "\n\n";

    // ----------- Safety check -----------
    std::cout << "Original tensor again (must be unchanged):\n";
    std::cout << tensS << "\n\n";
    // Optional sanity checks if operator() / at() exists
    // std::cout << "t(0,0,0) = " << t.at({0,0,0}) << "\n"; // 0
    // std::cout << "t(1,2,4) = " << t.at({1,2,4}) << "\n"; // 29


    std::vector<int> data_x(40);
    for (int i = 0; i < 40; ++i)
        data_x[i] = i;

    // 3D tensor: 5 x 4 x 2  → 40 elements
    std::vector<size_t> dimensions = {5, 4, 2};
    tensor<int> tensS(data_x, dimensions);

    std::cout << "Base tensor (5,4,2):\n";
    std::cout << tensS << "\n\n";

    // ---------------- Iteration 1: VIEW (neg, pos, neg) ----------------
    auto v1 = tensS.slice(
        Slice(-1, -6, -1), // axis 0: reverse
        Slice(0, 4, 2),    // axis 1: even columns
        Slice(-1, -3, -1)  // axis 2: reverse
    );

    std::cout << "v1 = slice_view:\n";
    std::cout << v1 << "\n\n";

    // ---------------- Iteration 2: COPY (pos, neg, pos) ----------------
    auto c2 = v1.slice(
        Slice(0, 3, 1),  // axis 0
        Slice(2, 0, -1), // axis 1
        Slice(0, 1, 1)   // axis 2
    );

    std::cout << "c2 = slice (from view):\n";
    std::cout << c2 << "\n\n";

    // ---------------- Iteration 3: VIEW (neg, neg, neg) ----------------
    auto v3 = c2.slice_view(
        Slice(-1, -4, -1),
        Slice(-1, -3, -1),
        Slice(-1, -2, -1));

    std::cout << "v3 = slice_view (from copy):\n";
    std::cout << v3 << "\n\n";

    // ---------------- Iteration 4: COPY (mixed) ----------------
    auto c4 = v3.slice(
        Slice(0, 2),
        Slice(1, -1, -1),
        Slice(0, 1));

    std::cout << "c4 = slice (from view):\n";
    std::cout << c4 << "\n\n";

    // ---------------- Safety checks ----------------
    std::cout << "Original tensor again (must be unchanged):\n";
    std::cout << tensS << "\n\n";

    std::cout << "Sanity values:\n";
    std::cout << "tensS(0,0,0) = " << tensS(0, 0, 0) << "\n";
    std::cout << "v1(0,0,0)    = " << v1(0, 0, 0) << "\n";
    std::cout << "c2(0,0,0)    = " << c2(0, 0, 0) << "\n";
    std::cout << "v3(0,0,0)    = " << v3(0, 0, 0) << "\n";

    std::vector<int> raw_data_200(200);
    for (int i = 0; i < 200; ++i)
        raw_data_200[i] = i;

    // ---------------- Dimensions ----------------
    // 5 × 8 × 5 = 200
    std::vector<size_t> dims_3d = {5, 8, 5};

    tensor<int> base_tensor(raw_data_200, dims_3d);

    std::cout << "Base tensor (5,8,5):\n";
    std::cout << base_tensor << "\n\n";

    // ============================================================
    // Iteration 1 — VIEW (neg, pos, neg)
    // ============================================================
    auto view_A = base_tensor.slice_view(
        Slice(-1, -6, -1), // axis 0: reverse [4,3,2,1,0]
        Slice(1, 8, 2),    // axis 1: [1,3,5,7]
        Slice(-1, -6, -1)  // axis 2: reverse
    );

    std::cout << "view_A:\n";
    std::cout << view_A << "\n\n";

    // ============================================================
    // Iteration 2 — COPY (pos, neg, pos)
    // ============================================================
    auto copy_B = view_A.slice(
        Slice(0, 3),     // axis 0: first 3
        Slice(3, 0, -1), // axis 1: reverse subset
        Slice(0, 3)      // axis 2: shrink
    );

    std::cout << "copy_B:\n";
    std::cout << copy_B << "\n\n";

    // ============================================================
    // Iteration 3 — VIEW (all negative)
    // ============================================================
    auto view_C = copy_B.slice_view(
        Slice(-1, -4, -1),
        Slice(-1, -5, -2),
        Slice(-1, -4, -1));

    std::cout << "view_C:\n";
    std::cout << view_C << "\n\n";

    // ============================================================
    // Iteration 4 — COPY (mixed, axis collapse)
    // ============================================================
    auto copy_D = view_C.slice(
        Slice(0, 2),      // axis 0
        Slice(1, -1, -1), // axis 1 (may shrink to 1)
        Slice(0, 1)       // axis 2 collapse
    );

    std::cout << "copy_D:\n";
    std::cout << copy_D << "\n\n";

    // ============================================================
    // Iteration 5 — VIEW (final permutation)
    // ============================================================
    auto view_E = copy_D.slice_view(
        Slice(-1, -3, -1),
        Slice(0, 2),
        Slice(-1, -2, -1));

    std::cout << "view_E:\n";
    std::cout << view_E << "\n\n";

    // ============================================================
    // Safety / sanity checks
    // ============================================================
    std::cout << "Original tensor (must be unchanged):\n";
    std::cout << base_tensor << "\n\n";

    std::cout << "Sanity values:\n";
    std::cout << "base_tensor(0,0,0) = " << base_tensor(0, 0, 0) << "\n";
    std::cout << "view_A(0,0,0)      = " << view_A(0, 0, 0) << "\n";
    std::cout << "copy_B(0,0,0)      = " << copy_B(0, 0, 0) << "\n";
    std::cout << "view_C(0,0,0)      = " << view_C(0, 0, 0) << "\n";
    std::cout << "copy_D(0,0,0)      = " << copy_D(0, 0, 0) << "\n";
    std::cout << "view_E(0,0,0)      = " << view_E(0, 0, 0) << "\n";

    std::vector<int> dataA = {
        1, 2, 3,
        4, 5, 6};
    std::vector<size_t> dimA = {2, 3};
    tensor<int> A(dataA, dimA);

    std::vector<int> dataB = {
        10, 20,
        30, 40,
        50, 60};
    std::vector<size_t> dimB = {3, 2};
    tensor<int> B(dataB, dimB);

    auto C = A.tensor_prod(B);
    std::cout << C << std::endl;
    std::cout << C.reshape({6, 6}) << std::endl;
    */
    /*
     std::vector<int> dataA = {
         1, 2,
         3, 4,
         5, 6};
     std::vector<size_t> dimA = {3, 2};
     tensor<int> A(dataA, dimA);

     // ---------- Base matrix B: 2 x 2 ----------
     std::vector<int> dataB = {
         7, 8,
         9, 10};
     std::vector<size_t> dimB = {2, 2};
     tensor<int> B(dataB, dimB);

     std::cout << "Base A:\n"
               << A << "\n\n";
     std::cout << "Base B:\n"
               << B << "\n\n";

     // ---------- Views with negative indices ----------
     // reverse rows AND columns

     auto vA = A.slice_view(
         Slice(-1, -4, -1), // rows reversed
         Slice(-1, -3, -1)  // cols reversed
     );

     auto vB = B.slice_view(
         Slice(-1, -3, -1),
         Slice(-1, -3, -1));

     std::cout << "View vA (reversed A):\n"
               << vA << "\n\n";
     std::cout << "View vB (reversed B):\n"
               << vB << "\n\n";

     // ---------- Tensor product ----------
     auto C = vA.tensor_prod(vB);

     std::cout << "C = vA tensor_prod vB:\n";
     std::cout << C << "\n";
     */
    std::vector<int> dataA = {1, 2};
    std::vector<int> dataB = {3, 4};
    std::vector<int> dataC = {5, 6};

    std::vector<size_t> dims = {2, 1, 1};

    tensor<int> A(dataA, dims);
    tensor<int> B(dataB, dims);
    tensor<int> C(dataC, dims);

    std::cout << "A:\n"
              << A << "\n\n";
    std::cout << "B:\n"
              << B << "\n\n";
    std::cout << "C:\n"
              << C << "\n\n";

    // ---------------- Views with mixed slicing ----------------
    // A: reverse first axis

    auto vonA = A.slice_view(
        Slice(-1, -3, -1), // axis 0 reversed
        Slice(0, 1),
        Slice(0, 1));

    // B: normal slice (positive)
    auto vonB = B.slice_view(
        Slice(0, 2),
        Slice(0, 1),
        Slice(0, 1));

    // C: reverse first axis
    auto vonC = C.slice_view(
        Slice(-1, -3, -1),
        Slice(0, 1),
        Slice(0, 1));

    std::cout << "vA:\n"
              << vonA << "\n\n";
    std::cout << "vB:\n"
              << vonB << "\n\n";
    std::cout << "vC:\n"
              << vonC << "\n\n";

    // ---------------- Triple tensor product ----------------
    // ((vA ⊗ vB) ⊗ vC)
    auto T = vonA.tensor_prod(vonB).tensor_prod(vonC);

    std::cout << "T = vA ⊗ vB ⊗ vC:\n";
    std::cout << T << std::endl;
    // std::cout << T << "\n";
    std::cout << "===== LARGE SCALE TENSOR STRESS TEST =====\n\n";

    // ============================================================
    // 1. Massive Base Tensor (≈ 2 million elements)
    // ============================================================

    constexpr size_t D0 = 200;
    constexpr size_t D1 = 200;
    constexpr size_t D2 = 50;
    constexpr size_t TOTAL_SIZE = D0 * D1 * D2;

    std::vector<int> massive_data(TOTAL_SIZE);
    for (size_t i = 0; i < TOTAL_SIZE; ++i)
        massive_data[i] = static_cast<int>(i % 97);

    std::vector<size_t> massive_dims = {D0, D1, D2};

    tensor<int> massive_tensor(massive_data, massive_dims);

    std::cout << "Massive tensor created: "
              << D0 << " x " << D1 << " x " << D2
              << " (" << TOTAL_SIZE << " elements)\n\n";

    // ============================================================
    // 2. Massive SLICE (deep copy)
    // ============================================================

    time_block("Massive slice (copy)", [&]()
               {
        auto big_slice_copy = massive_tensor.slice(
            Slice(-1, -101, -1),   // reverse first 100
            Slice(0, 200, 2),      // stride
            Slice(10, 40, 1)       // narrow
        );
        volatile int sink = big_slice_copy(0,0,0);
        (void)sink; });

    // ============================================================
    // 3. Massive SLICE_VIEW (no copy)
    // ============================================================

    time_block("Massive slice_view", [&]()
               {
        auto big_slice_view = massive_tensor.slice_view(
            Slice(-1, -101, -1),
            Slice(0, 200, 2),
            Slice(10, 40, 1)
        );
        volatile int sink = big_slice_view(0,0,0);
        (void)sink; });

    // ============================================================
    // 4. Repeated View → Copy → View Chain (worst traversal)
    // ============================================================

    time_block("Repeated view/copy chaining", [&]()
               {

        auto v1 = massive_tensor.slice_view(
            Slice(-1, -151, -1),
            Slice(0, 200, 3),
            Slice(0, 50)
        );

        auto c2 = v1.slice(
            Slice(10, 80),
            Slice(-1, -40, -2),
            Slice(5, 30)
        );

        auto v3 = c2.slice_view(
            Slice(-1, -20, -1),
            Slice(0, 10),
            Slice(-1, -10, -1)
        );

        volatile int sink = v3(0,0,0);
        (void)sink; });

    // ============================================================
    // 5. Tensor Product Stress (SAFE SIZE)
    // ============================================================

    // Reduce dimensionality to avoid RAM explosion
    std::vector<int> smallA_data(2000);
    std::vector<int> smallB_data(1500);

    for (size_t i = 0; i < smallA_data.size(); ++i)
        smallA_data[i] = i % 11;
    for (size_t i = 0; i < smallB_data.size(); ++i)
        smallB_data[i] = i % 13;

    tensor<int> smallA(smallA_data, {200, 10});
    tensor<int> smallB(smallB_data, {150, 10});

    time_block("Tensor product (2000 x 1500)", [&]()
               {
        auto prod = smallA.tensor_prod(smallB);
        volatile int sink = prod(0,0,0,0);
        (void)sink; });

    // ============================================================
    // 6. Triple Tensor Product (Views)
    // ============================================================

    auto vA1 = smallA.slice_view(Slice(-1, -201, -1), Slice(0, 10));
    auto vB1 = smallB.slice_view(Slice(0, 150), Slice(-1, -11, -1));

    std::vector<int> smallC_data(500);
    for (size_t i = 0; i < smallC_data.size(); ++i)
        smallC_data[i] = i % 7;

    tensor<int> smallC(smallC_data, {50, 10});
    auto vC1 = smallC.slice_view(Slice(0, 50), Slice(0, 10));

    time_block("Triple tensor product (views)", [&]()
               {
        auto triple = vA1.tensor_prod(vB1).tensor_prod(vC1);
        volatile int sink = triple(0,0,0,0,0,0);
        (void)sink; });

    std::cout << "\n===== STRESS TEST COMPLETE =====\n";
}

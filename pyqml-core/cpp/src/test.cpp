#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <complex.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::shuffle
#include <random>
#include <memory>

template <typename F>
void time_block(const std::string &name, F &&fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < 500; ++j)
        fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << name << ": "
              << std::chrono::duration<double>(t1 - t0).count() / 500
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
    /*
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
                     Slice(0, 1))
                    .copy();

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
        auto big_slice_copy = massive_tensor.slice_view(
            Slice(-1, -101, -1),   // reverse first 100
            Slice(0, 200, 2),      // stride
            Slice(10, 40, 1)       // narrow
        ).copy();
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

    constexpr size_t X = 256;
    constexpr size_t Y = 256;
    constexpr size_t Z = 64;
    constexpr size_t TOTAL = X * Y * Z;

    std::vector<int> raw(TOTAL);
    for (size_t i = 0; i < TOTAL; ++i)
        raw[i] = int(i % 97);

    tensor<int> big(raw, {X, Y, Z});

    std::cout << "Tensor: "
              << X << " x " << Y << " x " << Z
              << " (" << TOTAL << " elements)\n";

    // Warm‑up
    {
        auto tmp = big.slice_view(
                          Slice(0, X, 1),
                          Slice(0, Y, 2),
                          Slice(1, Z, 3))
                       .copy();
        volatile int sink = tmp(0, 0, 0);
        (void)sink;
    }

    // Timed
    time_block("C++ strided slice + copy", [&]()
               {
    auto out = big.slice_view(
        Slice(0, X, 2),   // contiguous
        Slice(0, Y, 4),   // stride 2
        Slice(1, Z, 3)    // stride 3  ❗ not contiguous
    ).copy();

    volatile int sink = out(0,0,0);
    (void)sink; });

    time_block("C++ strided slice_view ONLY", [&]()
               {
    auto view = big.slice_view(
        Slice(0, X, 2),   // contiguous
        Slice(0, Y, 4),   // stride 2
        Slice(1, Z, 3)    // stride 3 (non-contiguous)
    );

    volatile int sink = view(0, 0, 0);
    (void)sink; });
    */

    /*
     std::vector<int> data(200);
     for (int i = 0; i < 200; ++i)
         data[i] = i;

     tensor<int> T(data, {5, 4, 5, 2});

     std::cout << "Original tensor T (5,4,5,2):\n";
     std::cout << T << "\n\n";

     auto V = T.slice_view(
         Index(3),   // axis 0 → keep
         Index(2),   // axis 1 → REMOVE
         Index(4),   // axis 2 → keep
         Slice(0, 2) // axis 3 → REMOVE
     );

     std::cout << "View V (after 2 Index reductions):\n";
     std::cout << V << "\n\n";
     */
    /*
    std::vector<size_t> shape = {5, 5, 4, 2};

    std::vector<int> data(200);
    for (int i = 0; i < 200; ++i)
        data[i] = i;

    tensor<int> t(std::move(data), shape);

    std::cout << "Original tensor shape: ";
    for (auto d : t.shape())
        std::cout << d << " ";
    std::cout << "\n\n";
    */

    // ---- 2. Construct a *pathological* slicing pattern ----
    //
    // Axis 0: Range (non-affine, irregular)
    // Axis 1: Slice (non-unit stride)
    // Axis 2: Index (dimension reduction)
    // Axis 3: Range (non-affine, irregular)
    //
    // IMPORTANT: all Range indices are positive

    /*

    Range r0({0, 4, 2}); // irregular gather on axis 0
    Slice s1(1, 5, 2);   // slice [1:5:2] on axis 1
    Index i2(2);         // fix axis 2
    Range r3({1, 0});    // trivial but still Range on axis 3

    // ---- 3. Apply slice (forces general / radix path) ----
    auto out = t.slice(r0, s1, i2, r3);

    // ---- 4. Print result ----
    std::cout << "Sliced tensor shape: ";
    for (auto d : out.shape())
        std::cout << d << " ";
    std::cout << "\n\n";

    std::cout << "Sliced tensor contents:\n";
    std::cout << out << "\n";
    */
    /*
    constexpr size_t D0 = 128;
    constexpr size_t D1 = 192;
    constexpr size_t D2 = 64;
    constexpr size_t D3 = 32;

    constexpr size_t TOTAL = D0 * D1 * D2 * D3;

    // ---- Raw data ----
    std::vector<int> raw_big(TOTAL);
    for (size_t i = 0; i < TOTAL; ++i)
        raw_big[i] = int(i % 251);

    tensor<int> giant(raw_big, {D0, D1, D2, D3});

    std::cout << "Giant tensor shape: "
              << D0 << " x " << D1 << " x "
              << D2 << " x " << D3
              << " (" << TOTAL << " elements)\n";

    // ========================================================
    // Warm‑up (important for cache + branch predictor)
    // ========================================================
    {
        auto warm = giant.slice_view(
                             Slice(0, D0, 1),
                             Slice(0, D1, 2),
                             Slice(0, D2, 3),
                             Slice(0, D3, 4))
                        .copy();

        volatile int sink = warm(0, 0, 0, 0);
        (void)sink;
    }

    // ========================================================
    // Pathological axis definitions
    //  - Ranges are POSITIVE ONLY
    //  - Slices may be negative
    //  - Index may be negative
    // ========================================================
    /*
    Range rg0({0, 7, 3, 11, 2, 19}); // irregular gather
    Slice sl1(-1, -150, -2);         // reversed, strided
    Index ix2(-3);                   // negative index
    Range rg3({1, 4, 0, 7, 2});      // irregular gather

    // ========================================================
    // 1. Full pathological slice → view → slice chain
    // ========================================================

    time_block("4D pathological slice chain", [&]()
               {

        auto view0 = giant.slice(
            rg0,
            sl1,
            ix2,
            rg3
        );

        auto copy1 = view0.slice(
            Slice(0, 10),
            Slice(-1, -20, -1),
            Range({0, 2, 1, 3}),
            Slice(0, 5)
        );


        volatile int sink = copy1(0, 0, 0, 0);
        (void)sink; });
    */

    // ========================================================
    // 2. Same pathological slice — COPY
    // ========================================================
    /*
       time_block("4D pathological slice (copy)", [&]()
                  {

           auto out = giant.slice(
               rg0,
               sl1,
               ix2,
               rg3
           );

           volatile int sink = out(0, 0, 0);
           (void)sink; });

       // ========================================================
       // 3. Same pathological slice — VIEW
       // ========================================================

       time_block("4D pathological slice (view)", [&]()
                  {

           auto out = giant.slice(
               rg0,
               sl1,
               ix2,
               rg3
           );

           volatile int sink = out(0, 0, 0);
           (void)sink; });

       std::cout << "\n===== 4D PATHOLOGICAL STRESS TEST COMPLETE =====\n";
       */

    // ============================================================
    // 1. Massive Base Tensor (≈ 2 million elements)
    // ============================================================
    // ============================================================
    // 1. Massive Base Tensor (≈ 2 million elements)
    // ============================================================
    /*
    constexpr size_t D0 = 200;
    constexpr size_t D1 = 200;
    constexpr size_t D2 = 50;
    constexpr size_t TOTAL_SIZE = D0 * D1 * D2;

    std::vector<int32_t> massive_data(TOTAL_SIZE);
    for (size_t i = 0; i < TOTAL_SIZE; ++i)
        massive_data[i] = static_cast<int32_t>(i % 97);

    std::vector<size_t> massive_dims = {D0, D1, D2};

    tensor<int32_t> massive_tensor(std::move(massive_data), massive_dims);

    std::cout << "Massive tensor created: "
              << D0 << " x " << D1 << " x " << D2
              << " (" << TOTAL_SIZE << " elements)\n\n";

    // ============================================================
    // 2. Massive SLICE (deep copy)
    // ============================================================

    time_block("Massive slice (copy)", [&]()
               {
        tensor<int32_t> big_slice_copy = massive_tensor.slice_view(
            Slice(-1, -101, -1),   // reverse first 100
            Slice(0, 200, 2),      // stride
            Slice(10, 40, 1)       // narrow
        ).copy();
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

        tensor<int32_t> v1 = massive_tensor.slice_view(
            Slice(-1, -151, -1),
            Slice(0, 200, 3),
            Slice(0, 50)
        );

        tensor<int32_t> c2 = v1.slice_view(
            Slice(10, 80),
            Slice(-1, -40, -2),
            Slice(5, 30)
        ).copy();

        tensor<int32_t> v3 = c2.slice_view(
            Slice(-1, -20, -1),
            Slice(0, 10),
            Slice(-1, -10, -1)
        );

        volatile int sink = v3(0,0,0);
        (void)sink; });

    std::vector<size_t> dims = {1000, 1000};
    std::vector<size_t> dims2 = {1, 1000};
    size_t total_size = 1000 * 1000;
    size_t total_size2 = 1 * 1000;

    // fill large data arrays
    std::vector<double> data_a(total_size);
    std::vector<double> data_b(total_size2);

    // initialize with some values
    std::iota(data_a.begin(), data_a.end(), 0.0);    // 0,1,2,...
    std::iota(data_b.begin(), data_b.end(), 1000.0); // 1000,1001,...

    // create tensors
    tensor<double> a(std::move(data_a), dims);
    tensor<double> b(std::move(data_b), dims);

    // add
    // tensor<double> C = a + b;
    // std::cout << C;
    constexpr size_t BA = 512;
    constexpr size_t BB = 256;
    constexpr size_t BC = 64;

    constexpr size_t TOTAL_A = BA * BB * BC;

    // ---- Tensor A (512 x 256 x 64)
    std::vector<int_fast32_t> dataA(TOTAL_A);
    for (size_t i = 0; i < TOTAL_A; ++i)
        dataA[i] = int_fast32_t(i % 113);

    tensor<int_fast32_t> A(std::move(dataA), {BA, BB, BC});

    // ---- Tensor B (1 x 256 x 1)  -> broadcastable
    std::vector<int_fast32_t> dataB(BB);
    for (size_t i = 0; i < BB; ++i)
        dataB[i] = int_fast32_t(i % 17);

    tensor<int_fast32_t> B(std::move(dataB), {1, BB, 1});

    std::cout << "Broadcast test shapes:\n";
    std::cout << "A: 512 x 256 x 64\n";
    std::cout << "B: 1 x 256 x 1\n\n";

    /*
    // ---- Warm‑up
    {
        auto tmp = A + B;
        volatile int sink = tmp(0, 0, 0);
        (void)sink;
    }


    // ---- Timed

    time_block("Broadcast add (512x256x64) + (1x256x1)", [&]()
               {
    auto C = A-B;// - A/B*(A+B);
    volatile int sink = C(0,0,0);
    (void)sink; });

    std::vector<double> data_big_A(40 * 10 * 4);
    for (size_t i = 0; i < data_big_A.size(); ++i)
        data_big_A[i] = i;

    tensor<double> bigA(std::move(data_big_A), {10, 40, 4});

    // ---- Base tensor for B
    std::vector<double> data_big_B(40 * 20);
    for (size_t i = 0; i < data_big_B.size(); ++i)
        data_big_B[i] = i;

    tensor<double> bigB(std::move(data_big_B), {40, 10, 2});

    // ============================================================
    // Create slice_views
    // ============================================================

    // A slice → (2,5,4,2)
    auto A_V = bigA.slice_view(
        Slice(0, 1),   // 1
        Slice(20, 25), // 5
        Slice(0, 2)    // 2
    );

    // B slice → (2,5)
    auto B_V = bigB.slice_view(
        Slice(10, 15), // 5
        Slice(2, 3),   // 1
        Slice(0, 2)    // 2
    );

    std::cout << "Shapes:\n";
    std::cout << "A: (2,5,4,2)\n";
    std::cout << "B: (2,5)\n\n";

    // ============================================================
    // Broadcast add
    // ============================================================

    auto C_V = A_V - B_V;
    auto D_V = (A_V * B_V);
    // auto E_V = C_V - D_V;

    std::cout << "Result tensor:\n";
    std::cout << A_V << "\n\n";
    std::cout << B_V << "\n\n";
    std::cout << "C_V" << C_V << "\n\n";
    std::cout << "D_V" << D_V << "\n\n";
    // std::cout << "E_V" << E_V << "\n\n";
    */

    constexpr size_t D0 = 200;
    constexpr size_t D1 = 200;
    constexpr size_t D2 = 50;
    constexpr size_t TOTAL_SIZE = D0 * D1 * D2;

    auto massive_data = std::shared_ptr<int32_t[]>(new int32_t[TOTAL_SIZE]);
    for (size_t i = 0; i < TOTAL_SIZE; ++i)
        massive_data[i] = static_cast<int32_t>(i % 97);

    std::vector<size_t> massive_dims = {D0, D1, D2};

    tensor<int32_t> massive_tensor(massive_data, TOTAL_SIZE, massive_dims);

    std::cout << "Massive tensor created: "
              << D0 << " x " << D1 << " x " << D2
              << " (" << TOTAL_SIZE << " elements)\n\n";

    // ============================================================
    // 2. Massive SLICE (deep copy)
    // ============================================================

    time_block("Massive slice (copy)", [&]()
               {
    tensor<int32_t> big_slice_copy = massive_tensor.slice_view(
        Slice(-1, -101, -1),
        Slice(0, 200, 2),
        Slice(10, 40, 1)
    ).copy();

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
    // 4. Repeated View → Copy → View Chain
    // ============================================================

    time_block("Repeated view/copy chaining", [&]()
               {
    tensor<int32_t> v1 = massive_tensor.slice_view(
        Slice(-1, -151, -1),
        Slice(0, 200, 3),
        Slice(0, 50)
    );

    tensor<int32_t> c2 = v1.slice_view(
        Slice(10, 80),
        Slice(-1, -40, -2),
        Slice(5, 30)
    ).copy();

    tensor<int32_t> v3 = c2.slice_view(
        Slice(-1, -20, -1),
        Slice(0, 10),
        Slice(-1, -10, -1)
    );

    volatile int sink = v3(0,0,0);
    (void)sink; });

    std::vector<size_t> dims = {1000, 1000};
    std::vector<size_t> dims2 = {1, 1000};

    size_t total_size = 1000 * 1000;
    size_t total_size2 = 1 * 1000;

    auto data_a = std::shared_ptr<double[]>(new double[total_size]);
    auto data_b = std::shared_ptr<double[]>(new double[total_size2]);

    for (size_t i = 0; i < total_size; ++i)
        data_a[i] = static_cast<double>(i);

    for (size_t i = 0; i < total_size2; ++i)
        data_b[i] = static_cast<double>(1000 + i);

    tensor<double> a(data_a, total_size, dims);
    tensor<double> b(data_b, total_size2, dims2);

    // ============================================================
    // Broadcast Test
    // ============================================================

    constexpr size_t BA = 512;
    constexpr size_t BB = 256;
    constexpr size_t BC = 64;

    constexpr size_t TOTAL_A = BA * BB * BC;

    auto dataA = std::shared_ptr<int_fast32_t[]>(new int_fast32_t[TOTAL_A]);

    for (size_t i = 0; i < TOTAL_A; ++i)
        dataA[i] = int_fast32_t(i % 113);

    tensor<int_fast32_t> A(dataA, TOTAL_A, {BA, BB, BC});

    auto dataB = std::shared_ptr<int_fast32_t[]>(new int_fast32_t[BB]);

    for (size_t i = 0; i < BB; ++i)
        dataB[i] = int_fast32_t(i % 17);

    tensor<int_fast32_t> B(dataB, BB, {1, BB, 1});

    std::cout << "Broadcast test shapes:\n";
    std::cout << "A: 512 x 256 x 64\n";
    std::cout << "B: 1 x 256 x 1\n\n";

    time_block("Broadcast add (512x256x64) + (1x256x1)", [&]()
               {
    auto C = A - B;

    volatile int sink = C(0,0,0);
    (void)sink; });

    // ============================================================
    // Slice view broadcasting test
    // ============================================================

    constexpr size_t SIZE_A = 40 * 10 * 4;
    auto data_big_A = std::shared_ptr<double[]>(new double[SIZE_A]);

    for (size_t i = 0; i < SIZE_A; ++i)
        data_big_A[i] = i;

    tensor<double> bigA(data_big_A, SIZE_A, {10, 40, 4});

    constexpr size_t SIZE_B = 40 * 20;
    auto data_big_B = std::shared_ptr<double[]>(new double[SIZE_B]);

    for (size_t i = 0; i < SIZE_B; ++i)
        data_big_B[i] = i;

    tensor<double> bigB(data_big_B, SIZE_B, {40, 10, 2});

    // ============================================================
    // Create slice_views
    // ============================================================

    auto A_V = bigA.slice_view(
        Slice(0, 1),
        Slice(20, 25),
        Slice(0, 2));

    auto B_V = bigB.slice_view(
        Slice(10, 15),
        Slice(2, 3),
        Slice(0, 2));

    std::cout << "Shapes:\n";
    std::cout << "A: (2,5,4,2)\n";
    std::cout << "B: (2,5)\n\n";

    // ============================================================
    // Broadcast operations
    // ============================================================

    auto C_V = A_V - B_V;
    auto D_V = (A_V * B_V);

    std::cout << "Result tensor:\n";

    std::cout << A_V << "\n\n";
    std::cout << "Arg_max along axis 0 " << A_V.argmax(1) << std::endl;
    std::cout << B_V << "\n\n";

    std::cout << "C_V" << C_V << "\n\n";
    std::cout << "D_V" << D_V << "\n\n";
}

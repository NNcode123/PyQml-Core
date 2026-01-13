import numpy as np

import time

def time_block(name, fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    print(f"{name}: {t1 - t0:.6f} sec")
    return out


# -------------------------------
# Base tensors (similar sizes)
# -------------------------------
A = np.arange(2000, dtype=np.int32).reshape(200, 10)
B = np.arange(1500, dtype=np.int32).reshape(150, 10)
C = np.arange(500,  dtype=np.int32).reshape(50, 10)

# -------------------------------
# Create pathological *views*
# (negative strides, slicing)
# -------------------------------
vA = A[::-1, :]          # reverse axis 0
vB = B[:, ::-1]          # reverse axis 1
vC = C[::1, :]           # normal view (kept for comparison)

# sanity check: these are views
assert not vA.flags['C_CONTIGUOUS']
assert not vB.flags['C_CONTIGUOUS']

print("vA contiguous?", vA.flags['C_CONTIGUOUS'])
print("vB contiguous?", vB.flags['C_CONTIGUOUS'])
print("vC contiguous?", vC.flags['C_CONTIGUOUS'])


# -------------------------------
# Triple tensor product
# -------------------------------
def triple_tensor_product():
    # First outer product
    T1 = np.multiply.outer(vA, vB)
    # Second outer product
    T2 = np.multiply.outer(T1, vC)
    return T2


# -------------------------------
# Timing
# -------------------------------
T = time_block("Triple tensor product (NumPy)", triple_tensor_product)

# Touch one element so Python can't optimize it away
print("Sample:", T[0, 0, 0, 0, 0, 0])

D0 = 200
D1 = 200
D2 = 50
TOTAL_SIZE = D0 * D1 * D2

massive_data = np.arange(TOTAL_SIZE, dtype=np.int32) % 97
massive_tensor = massive_data.reshape(D0, D1, D2)

print(f"Massive tensor created: {D0} x {D1} x {D2} ({TOTAL_SIZE} elements)\n")


def chained_ops():
    v1 = massive_tensor[
        -1:-151:-1,
        0:200:3,
        0:50
    ]                  # view

    c2 = v1[
        10:80,
        -1:-40:-2,
        5:30
    ].copy()           # force copy

    v3 = c2[
        -1:-20:-1,
        0:10,
        -1:-10:-1
    ]                  # view again

    _ = v3[0, 0, 0]    # touch


time_block("Repeated view/copy chaining", chained_ops)

time_block("Massive slice (copy)", lambda: (
    lambda x: x[0, 0, 0]  # touch
)(
    massive_tensor[
        -1:-101:-1,   # reverse first 100
        0:200:2,      # stride
        10:40:1       # narrow
    ].copy()          # force deep copy
))


# ============================================================
# 3. Massive SLICE_VIEW (no copy)
# ============================================================

time_block("Massive slice_view", lambda: (
    lambda x: x[0, 0, 0]
)(
    massive_tensor[
        -1:-101:-1,
        0:200:2,
        10:40:1
    ]                  # view only
))

A = np.arange(200 * 10, dtype=np.int32).reshape(200, 10)
B = np.arange(150 * 10, dtype=np.int32).reshape(150, 10)

# Warm-up (important for fair timing)
_ = np.multiply.outer(A, B)

start = time.perf_counter()
prod = np.multiply.outer(A, B)
_ = prod[0, 0, 0, 0]   # touch
end = time.perf_counter()

print("NumPy tensor product time:", end - start)
print("Result shape:", prod.shape)

X, Y, Z = 256, 256, 64
TOTAL = X * Y * Z

data = (np.arange(TOTAL, dtype=np.int32) % 97).reshape(X, Y, Z)

print(f"Tensor: {X} x {Y} x {Z} ({TOTAL} elements)")

# Warm‑up
tmp = data[
    0:X:1,
    0:Y:2,
    1:Z:3
].copy()
_ = tmp[0,0,0]

# Timed
time_block("NumPy strided slice + copy", lambda: (
    lambda x: x[0,0,0]
)(
    data[
        0:X:1,   # contiguous
        0:Y:2,   # stride 2
        1:Z:3    # stride 3  ❗ not contiguous
    ].copy()
))

def numpy_slice_view_only():
    v = data[
        0:X:1,
        0:Y:2,
        1:Z:3
    ]
    _ = v[0, 0, 0]

time_block("NumPy strided slice_view ONLY", numpy_slice_view_only)

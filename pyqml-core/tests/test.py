
import numpy as np


import pyqmlcore as pyq


tens_1 = pyq.Tensor([3,4,5],[3,1])
tens_2 =tens_1
print(tens_2)








import time

def time_block(name, fn):
    t0 = time.perf_counter()
    for i in range (0, 1000):
        out = fn()   
    t1 = time.perf_counter()
    print(f"{name}: {(t1 - t0)/1000:.6f} sec")
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
"""
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
"""

# Touch one element so Python can't optimize it away
#print("Sample:", T[0, 0, 0, 0, 0, 0])


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
        0:X:2,   # contiguous
        0:Y:4,   # stride 2
        1:Z:3    # stride 3  ❗ not contiguous
    ].copy()
))

def numpy_slice_view_only():
    v = data[
        0:X:2,
        0:Y:2,
        1:Z:3
    ]
    _ = v[0, 0, 0]

time_block("NumPy strided slice_view ONLY", numpy_slice_view_only)



"""
D0 = 160
D1 = 200
D2 = 100
D3 = 8

TOTAL_ELEMS = D0 * D1 * D2 * D3

base_data = (np.arange(TOTAL_ELEMS, dtype=np.int32) % 113)
np_tensor = base_data.reshape(D0, D1, D2, D3)

print(f"Base NumPy tensor: {D0} x {D1} x {D2} x {D3} ({TOTAL_ELEMS} elements)")

# ============================================================
# Axis 0: RANDOM DISTINCT GATHER [0, 50]
# ============================================================

rng = np.random.default_rng(123)
axis0_indices = np.arange(51)
rng.shuffle(axis0_indices)
axis0_indices = axis0_indices[:50]     # distinct, unordered

# ============================================================
# Single pathological slice
# ============================================================
# Axis 0: advanced indexing (non-affine gather)
# Axis 1: negative-stride slice
# Axis 2: index (dimension drop)
# Axis 3: strided slice
#
# Expected output size:
#   50 * 100 * 4 = 20,000 elements

def numpy_pathological_slice():
    out = np_tensor[
        axis0_indices,        # axis 0 (gather)
        -1:-201:-2,           # axis 1 (negative stride)
        axis0_indices,     # axis 2 (index → drop dim)
        0:8:2                 # axis 3
    ]
    # touch one element so it can't be optimized away
    _ = out[0, 0, 0]
    return out

result = time_block(
    "NumPy single pathological slice (gather + slice + index)",
    numpy_pathological_slice
)

print("Output shape:", result.shape)
print("Output size:", result.size)
print("Contiguous?", result.flags['C_CONTIGUOUS'])

A = np.arange(2000, dtype=np.int32).reshape(200, 10) % 11
B = np.arange(1500, dtype=np.int32).reshape(150, 10) % 13

def time_block(name, fn):
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    print(f"{name}: {(t1 - t0)*1000:.2f} ms")

time_block("Tensor product", lambda: A[:, None, :, None] * B[None, :, None, :])

# Triple product
C = np.arange(500, dtype=np.int32).reshape(50, 10) % 7
time_block(
    "Triple tensor product",
    lambda: A[:,None,:,None,None,None]
          * B[None,:,None,:,None,None]
          * C[None,None,None,None,:,:]
)
"""

X = 512
Y = 256
Z = 64

# -----------------------------
# Create tensors
# -----------------------------
A = np.arange(X * Y * Z, dtype=int).reshape(X, Y, Z) % 97
B = np.arange(Y, dtype=int).reshape(1, Y, 1) % 31

print("Shapes:")
print("A:", A.shape)
print("B:", B.shape)
print()

# -----------------------------
# Warm-up (important!)
# -----------------------------
_ = A + B

# -----------------------------
# Timed broadcast add
# -----------------------------
C = time_block(
    "Broadcast add (512x256x64) + (1x256x1)",
    lambda: A - B
)

# Force usage so it isn't optimized away
_ = C[0,0,0]
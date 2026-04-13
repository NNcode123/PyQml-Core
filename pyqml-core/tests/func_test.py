import pyqmlcore as pyq
import numpy as np

# dims
D0, D1, D2 = 50, 40, 30   # smaller so you can debug easier

# build raw lists
li1 = [i % 97 for i in range(D0 * D1 * D2)]
li2 = [(i * 3) % 89 for i in range(D0 * D1 * D2)]

# build pyq tensors directly
tens_1 = pyq.Tensor(li1, dim=[D0, D1, D2], type=pyq.int64)
tens_2 = pyq.Tensor(li2, dim=[D0, D1, D2], type=pyq.int64)

# build numpy ONLY for validation
np_A = np.array(li1, dtype=np.int64).reshape(D0, D1, D2)
np_B = np.array(li2, dtype=np.int64).reshape(D0, D1, D2)

# -----------------------------
# 🔹 elementwise tests
# -----------------------------
add_pyq = tens_1 + tens_2
mul_pyq = tens_1 * tens_2

print("Add correct:",
      np.allclose(pyq.to_numpy(add_pyq), np_A + np_B))

print("Mul correct:",
      np.allclose(pyq.to_numpy(mul_pyq), np_A * np_B))

# -----------------------------
# 🔹 einsum test 1
# (i,j,k),(i,j,k)->(i)
# -----------------------------
axes_1 = [1, 2]
axes_2 = [1, 2]

res_pyq = pyq.einsum(tens_1, tens_2, axes_1, axes_2)
res_np  = np.einsum('ijk,ijk->i', np_A, np_B)

print("Einsum test 1 correct:",
      np.allclose(pyq.to_numpy(res_pyq), res_np))

# -----------------------------
# 🔹 einsum test 2 (axis reorder)
# (i,j,k),(k,j,l)->(i,l)
# -----------------------------
# build second tensor manually (NO numpy conversion)
D3 = 20
li3 = [(i * 5) % 83 for i in range(D2 * D1 * D3)]
tens_3 = pyq.Tensor(li3, dim=[D2, D1, D3], type=pyq.int64)

np_C = np.array(li3, dtype=np.int64).reshape(D2, D1, D3)

axes_1 = [1, 2]  # j,k
axes_2 = [1, 0]  # j,k (reordered)

res_pyq2 = pyq.einsum(tens_1, tens_3, axes_1, axes_2)
res_np2  = np.einsum('ijk,kjl->il', np_A, np_C)

print("Einsum test 2 correct:",
      np.allclose(pyq.to_numpy(res_pyq2), res_np2))

# -----------------------------
# 🔹 small debug print
# -----------------------------
print("Sample output (first 10):")
print(pyq.to_numpy(res_pyq)[:10])
print(res_np[:10])

import numpy as np
import time
import pyqmlcore as pyq
import matplotlib.pyplot as plt


"""
def time_block(name, fn):
    t0 = time.perf_counter()
    for i in range(0, 100):
        out = fn()
    t1 = time.perf_counter()
    print(f"{name}: {(t1 - t0)/100:.6f} sec")
    return (t1 - t0)/100

def pyq_test(shape_1, shape_2, op):
    size_1 = np.prod(shape_1)
    size_2 = np.prod(shape_2)

    data_1 = [i % 97 for i in range(size_1)]
    data_2 = [(i * 3) % 89 for i in range(size_2)]

    t1 = pyq.Tensor(data_1, dim=list(shape_1))
    t2 = pyq.Tensor(data_2, dim=list(shape_2))

    if op == "add":
        return t1 + t2
    if op == "sub":
        return t1 - t2
    if op == "mul":
        return t1 * t2
    if op == "einsum":
        return pyq.einsum(t1, t2, [len(shape_1)-1], [len(shape_2)-2])

def numpy_test(shape_1, shape_2, op):
    a = np.arange(np.prod(shape_1)).reshape(shape_1) % 97
    b = (np.arange(np.prod(shape_2)) * 3 % 89).reshape(shape_2)

    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "einsum":
        return np.matmul(a, b)

ops = ["add", "sub", "mul"]

broadcast_shapes = [
    ((64,64,64), (1,64,1)),
    ((128,128,64), (1,128,1)),
    ((128,128,128), (1,128,1))
]

einsum_shapes = [
    ((64,64), (64,64)),
    ((128,128), (128,128)),
    ((256,256), (256,256))
]

results_pyq = []
results_np = []
labels = []

for op in ops:
    for s1, s2 in broadcast_shapes:
        pyq_time = time_block(f"pyq_{op}_{s1}", lambda: pyq_test(s1, s2, op))
        np_time = time_block(f"np_{op}_{s1}", lambda: numpy_test(s1, s2, op))

        pyq_res = pyq.to_numpy(pyq_test(s1, s2, op))
        np_res = numpy_test(s1, s2, op)

        assert np.allclose(pyq_res, np_res)

        results_pyq.append(pyq_time)
        results_np.append(np_time)
        labels.append(f"{op}_{s1}")

for s1, s2 in einsum_shapes:
    pyq_time = time_block(f"pyq_einsum_{s1}", lambda: pyq_test(s1, s2, "einsum"))
    np_time = time_block(f"np_einsum_{s1}", lambda: numpy_test(s1, s2, "einsum"))

    pyq_res = pyq.to_numpy(pyq_test(s1, s2, "einsum"))
    np_res = numpy_test(s1, s2, "einsum")

    assert np.allclose(pyq_res, np_res)

    results_pyq.append(pyq_time)
    results_np.append(np_time)
    labels.append(f"einsum_{s1}")

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(14,7))
plt.bar(x - width/2, results_pyq, width, label="pyq")
plt.bar(x + width/2, results_np, width, label="numpy")

plt.xticks(x, labels, rotation=45)
plt.ylabel("Time (sec)")
plt.title("pyq vs numpy performance (scaled)")
plt.legend()
plt.tight_layout()
plt.savefig("results.png")
"""

import numpy as np
import time
import pyqmlcore as pyq
import matplotlib.pyplot as plt


def time_block(name, fn):
    t0 = time.perf_counter()
    for _ in range(1000):
        out = fn()
    t1 = time.perf_counter()
    print(f"{name}: {(t1 - t0)/1000:.6f} sec")
    return (t1 - t0)/1000

# ---------------- PREBUILD ----------------
def build_pyq(shape_1, shape_2):
    size_1 = np.prod(shape_1)
    size_2 = np.prod(shape_2)

    data_1 = [i % 97 for i in range(size_1)]
    data_2 = [(i * 3) % 89 for i in range(size_2)]

    t1 = pyq.Tensor(data_1, dim=list(shape_1))
    t2 = pyq.Tensor(data_2, dim=list(shape_2))
    return t1, t2

def build_numpy(shape_1, shape_2):
    a = np.arange(np.prod(shape_1)).reshape(shape_1) % 97
    b = (np.arange(np.prod(shape_2)) * 3 % 89).reshape(shape_2)
    return a, b

# ---------------- OPS ONLY ----------------
def pyq_op(t1, t2, op):
    if op == "add": return t1 + t2
    if op == "sub": return t1 - t2
    if op == "mul": return t1 * t2
    if op == "div": return t1/ t2
    if op == "einsum":
        return pyq.einsum(t1, t2, [len(t1.shape)-1], [len(t2.shape)-2])

def numpy_op(a, b, op):
    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "div": return a/b
    if op == "einsum": return np.matmul(a, b)

# ---------------- TEST ----------------
ops = ["add", "sub", "mul"]

broadcast_shapes = [
    ((64,64,64), (1,64,1)),
    ((128,128,64), (1,128,1)),
    ((128,128,128), (1,128,1))
]

einsum_shapes = [
    ((64,64), (64,64)),
    ((128,128), (128,128)),
    ((256,256), (256,256))
]

results_pyq = []
results_np = []
labels = []

# -------- broadcast --------
for op in ops:
    for s1, s2 in broadcast_shapes:
        t1, t2 = build_pyq(s1, s2)
        a, b = build_numpy(s1, s2)

        pyq_time = time_block(f"pyq_{op}_{s1}", lambda: pyq_op(t1, t2, op))
        np_time  = time_block(f"np_{op}_{s1}", lambda: numpy_op(a, b, op))

        assert np.allclose(pyq.to_numpy(pyq_op(t1, t2, op)), numpy_op(a, b, op))

        results_pyq.append(pyq_time)
        results_np.append(np_time)
        labels.append(f"{op}_{s1}")

# -------- einsum --------
for s1, s2 in einsum_shapes:
    t1, t2 = build_pyq(s1, s2)
    a, b = build_numpy(s1, s2)

    pyq_time = time_block(f"pyq_einsum_{s1}", lambda: pyq_op(t1, t2, "einsum"))
    np_time  = time_block(f"np_einsum_{s1}", lambda: numpy_op(a, b, "einsum"))

    assert np.allclose(pyq.to_numpy(pyq_op(t1, t2, "einsum")), numpy_op(a, b, "einsum"))

    results_pyq.append(pyq_time)
    results_np.append(np_time)
    labels.append(f"einsum_{s1}")

# ---------------- PLOT ----------------
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(14,7))
plt.bar(x - width/2, results_pyq, width, label="pyq")
plt.bar(x + width/2, results_np, width, label="numpy")

plt.xticks(x, labels, rotation=45)
plt.ylabel("Time (sec)")
plt.title("pyq vs numpy (ops only)")
plt.legend()
plt.tight_layout()
plt.savefig("results.png")
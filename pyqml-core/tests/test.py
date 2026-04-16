
import numpy as np
import time
import pyqmlcore as pyq
import matplotlib.pyplot as plt

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
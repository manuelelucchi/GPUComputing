from benchmark import gpu, cpu, cpu_iter
from matplotlib import pyplot as plt
import time

# Cache and measure jit compilation
e1 = gpu(5)
e2 = gpu(5)

print(f"Time for first compilation is {e1 - e2} ms")

x = []
y_iter = []
y_rec = []
y_gpu = []

for e in range(8, 20):
    n = 2**e
    iter = cpu_iter(n)
    rec = cpu(n)
    g = gpu(n)
    x.append(e)
    y_iter.append(iter)
    y_rec.append(rec)
    y_gpu.append(g)

plt.plot(x, y_iter, label="Iterative")
plt.plot(x, y_rec, label="Recursive")
plt.plot(x, y_gpu, label="Numba")
plt.legend()
plt.xlabel("Size of the input (in power of 2)")
plt.ylabel("Time (s)")
plt.savefig("./report/images/cpu_vs_gpu.png")
plt.close()

x = list(range(15, 28))
y_c = [
    0.98435,
    1.03830,
    1.47274,
    1.88112,
    2.81171,
    8.38560,
    16.29379,
    35.05082,
    69.36614,
    147.84259,
    320.63742,
    671.53595,
    1471.70715,
]
y_py = [gpu(2**e) * 1000 for e in x]

print(y_c)
print(y_py)

plt.title("CUDA C vs Numba")
plt.plot(x, y_c, label="CUDA C")
plt.plot(x, y_py, label="Numba")
plt.xlabel("Size of the input (in power of 2)")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig("./report/images/cpp_vs_py.png")

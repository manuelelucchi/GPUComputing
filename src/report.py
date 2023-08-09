from benchmark import gpu, cpu, cpu_iter
from matplotlib import pyplot as plt

x = []
y_iter = []
y_rec = []
y_gpu = []

for e in range(8, 15):
    n = 2**e
    iter = cpu_iter(n)
    rec = cpu(n)
    g = gpu(n)
    x.append(n)
    y_iter.append(iter)
    y_rec.append(rec)
    y_gpu.append(g)

plt.plot(x, y_iter)
plt.plot(x, y_rec)
plt.plot(x, y_gpu)
plt.savefig("./report/images/cpu_vs_gpu.png")
plt.show()

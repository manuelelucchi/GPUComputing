import bitonic_cpu
import bitonic_gpu
from time import time


def cpu(n):
    data = bitonic_cpu.generate(n)

    start = time()
    bitonic_cpu.bitonic_sort(data, n, 0)
    end = time()

    print(f"Sorted {n} elements in {end - start}")
    return end - start


def cpu_iter(n):
    data = bitonic_cpu.generate(n)

    start = time()
    bitonic_cpu.bitonic_sort_iter(data, n, 0)
    end = time()

    print(f"Sorted {n} elements in {end - start}")
    return end - start


def gpu(n):
    data_gpu = bitonic_gpu.generate(n)

    start = time()
    bitonic_gpu.bitonic_sort(data_gpu, n, 0)
    end = time()

    print(f"Sorted {n} elements in {end - start}")
    return end - start


# https://numba.pydata.org/numba-doc/latest/cuda/memory.html


def cpu_bench():
    for n in [pow(2, i) for i in range(8, 20)]:
        cpu(n)


def gpu_bench():
    for n in [pow(2, i) for i in range(8, 25)]:
        gpu(n)

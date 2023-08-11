import bitonic_cpu
import bitonic_gpu
import time


def cpu(n):
    data = bitonic_cpu.generate(n)

    start = time.perf_counter()
    bitonic_cpu.bitonic_sort(data, n, 0)
    end = time.perf_counter()

    return end - start


def cpu_iter(n):
    data = bitonic_cpu.generate(n)

    start = time.perf_counter()
    bitonic_cpu.bitonic_sort_iter(data, n, 0)
    end = time.perf_counter()

    return end - start


def gpu(n):
    data_gpu = bitonic_gpu.generate(n)

    s = time.perf_counter()
    data_gpu.copy_to_host()
    e = time.perf_counter()

    start = time.perf_counter()
    bitonic_gpu.bitonic_sort(data_gpu, n, 0)
    data_gpu.copy_to_host()
    end = time.perf_counter()

    return (end - start) - (e - s)


# https://numba.pydata.org/numba-doc/latest/cuda/memory.html


def cpu_bench():
    for n in [pow(2, i) for i in range(8, 20)]:
        cpu(n)


def gpu_bench():
    for n in [pow(2, i) for i in range(8, 25)]:
        gpu(n)

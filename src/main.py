import bitonic_gpu
import bitonic_cpu
from time import time

# 0 = descending
# 1 = ascending


def test_gpu(n, direction):
    data_gpu = bitonic_gpu.generate(n)

    print("Data before sorting")
    print(data_gpu.copy_to_host())

    start = time()
    bitonic_gpu.bitonic_sort(data_gpu, n, direction)

    print("Sorted data")
    print(data_gpu.copy_to_host())
    end = time()
    print(f"Time {end - start}")


def test_cpu_rec(n, direction):
    data = bitonic_cpu.generate(n)

    print("Data before sorting")
    print(data)

    start = time()
    bitonic_cpu.bitonic_sort(data, n, direction)
    end = time()

    print("Sorted data")
    print(data)
    print(f"Time {end - start}")


def test_cpu_iter(n, direction):
    data = bitonic_cpu.generate(n)

    print("Data before sorting")
    print(data)

    start = time()
    bitonic_cpu.bitonic_sort_iter(data, n, direction)
    end = time()

    print("Sorted data")
    print(data)
    print(f"Time {end - start}")


n = 2**28
direction = 0

# test_cpu_rec(n, direction)
# test_cpu_iter(n, direction)
test_gpu(n, direction)

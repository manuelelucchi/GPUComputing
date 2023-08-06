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
    bitonic_gpu.bitonic_sort(data_gpu, n)
    end = time()

    print("Sorted data")
    print(data_gpu.copy_to_host())
    print(f"In {end - start}")


def test_cpu(n, direction):
    data = bitonic_cpu.generate(n)

    print("Data before sorting")
    print(data)

    start = time()
    bitonic_cpu.bitonic_sort(data, n, direction)
    end = time()

    print("Sorted data")
    print(data)
    print(f"In {end - start}")


test_cpu(32768, 1)

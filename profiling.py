# https://numba-how-to.readthedocs.io/en/latest/profiling.html

import bitonic_gpu
from time import time

n = pow(2, 15)

data_gpu = bitonic_gpu.generate(n)

start = time()
bitonic_gpu.bitonic_sort(data_gpu, n)
end = time()

print(f"Sorted {n} elements in {end - start}")

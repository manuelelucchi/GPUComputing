from numba import cuda
import numpy as np
from utils import compare_and_swap_device
import time

threads = 256


def generate(n):
    out = np.random.rand(n)
    out_gpu = cuda.to_device(out)
    return out_gpu


@cuda.jit  # (nb.)
def bitonic_kernel(data, j, k, direction):
    i = cuda.grid(1)
    ixj = i ^ j

    # USARE CAS https://numba.readthedocs.io/en/stable/cuda/intrinsics.html
    if ixj > i:
        if (i & k) == 0:
            compare_and_swap_device(data, i, ixj, direction)

        if (i & k) != 0:
            compare_and_swap_device(data, ixj, i, direction)


def bitonic_sort(data, n, direction):
    blocks = (n + threads - 1) // threads
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            bitonic_kernel[blocks, threads](data, j, k, direction)
            j >>= 1
        k <<= 1
    return time.perf_counter()


# https://github.com/nschloe/perfplot

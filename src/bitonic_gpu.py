from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
from utils import compare_and_swap_device

threads = 256


@cuda.jit
def generate_kernel(n, states, out):
    thread_id = cuda.grid(1)
    if thread_id < n:
        out[thread_id] = xoroshiro128p_uniform_float32(states, thread_id)


def generate(n):
    blocks = (n + threads - 1) // threads
    states = create_xoroshiro128p_states(threads * blocks, seed=1)
    out = np.zeros(threads * blocks, dtype=np.float32)
    out_gpu = cuda.to_device(out)
    generate_kernel[blocks, threads](n, states, out_gpu)
    return out_gpu


@cuda.jit
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

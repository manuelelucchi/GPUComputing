import numpy as np
from utils import compare_and_swap


def generate(n):
    return np.random.random(n)


def bitonic_merge(data, low, n, direction):
    if n > 1:
        k = n // 2
        for i in range(low, low + k):
            compare_and_swap(data, i, i + k, direction)
        bitonic_merge(data, low, k, direction)
        bitonic_merge(data, low + k, k, direction)


def _bitonic_sort(data, low, n, direction):
    if n > 1:
        k = n // 2
        _bitonic_sort(data, low, k, not direction)
        _bitonic_sort(data, low + k, k, direction)
        bitonic_merge(data, low, n, direction)


def bitonic_sort(data, n, direction):
    _bitonic_sort(data, 0, n, direction)


def bitonic_sort_iter(data, n, direction):
    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            for i in range(0, n):
                l = i ^ j
                if l > i:
                    if i & k == 0:
                        compare_and_swap(data, i, l, direction)
                    if i & k != 0:
                        compare_and_swap(data, l, i, direction)
            j //= 2
        k *= 2

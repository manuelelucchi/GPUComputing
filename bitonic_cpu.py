import numpy as np


def generate(n):
    return np.random.random(n)


def compare_and_swap(data, i, j, direction):
    if direction == (data[i] > data[j]):
        data[i], data[j] = data[j], data[i]


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
        _bitonic_sort(data, low, k, 1)
        _bitonic_sort(data, low + k, k, 0)
        bitonic_merge(data, low, n, direction)


def bitonic_sort(data, n, direction):
    _bitonic_sort(data, 0, n, direction)

from numba import cuda


@cuda.jit
def greatest_power_of_two_less_than_device(n):
    k = 1
    while k > 0 and k < n:
        k = k << 1
    return k >> 1


def greatest_power_of_two_less_than(n):
    k = 1
    # 2^n < a <= 2^(n+1)
    while k > 0 and k < n:
        k = k << 1
    # // return 2^n
    return k >> 1


@cuda.jit
def compare_and_swap_device(data, i, j, direction):
    if direction == (data[i] > data[j]):
        data[i], data[j] = data[j], data[i]


def compare_and_swap(data, i, j, direction):
    if direction == (data[i] > data[j]):
        data[i], data[j] = data[j], data[i]

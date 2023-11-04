from numba import cuda


@cuda.jit
def compare_and_swap_device(data, i, j, direction):
    if direction == (data[i] > data[j]):
        data[i], data[j] = data[j], data[i]


def compare_and_swap(data, i, j, direction):
    if direction == (data[i] > data[j]):
        data[i], data[j] = data[j], data[i]

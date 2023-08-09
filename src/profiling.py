# https://numba-how-to.readthedocs.io/en/latest/profiling.html

import bitonic_gpu
import numba

n = pow(2, 15)

data_gpu = bitonic_gpu.generate(n)

bitonic_gpu.bitonic_sort(data_gpu, n, 0)

numba.cuda.profile_stop()

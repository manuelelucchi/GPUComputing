#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define THREADS 256
#define BLOCKS 8 * 1024

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
        }                                                          \
    }

__global__ void bitonic_sort_step(int *a, int j, int k)
{
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    if ((ixj) > i)
    {
        if ((i & k) == 0)
        {
            if (a[i] > a[ixj])
            {
                int temp = a[i];
                a[i] = a[ixj];
                a[ixj] = temp;
            }
        }
        if ((i & k) != 0)
        {
            if (a[i] < a[ixj])
            {
                int temp = a[i];
                a[i] = a[ixj];
                a[ixj] = temp;
            }
        }
    }
}

int main(void)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = THREADS * BLOCKS;
    // check
    if (!(N && !(N & (N - 1))))
    {
        printf("ERROR: N must be power of 2 (N = %d)\n", N);
        exit(1);
    }
    size_t nBytes = N * sizeof(int);
    int *a = (int *)malloc(nBytes);
    int *b = (int *)malloc(nBytes);

    // fill data
    for (int i = 0; i < N; ++i)
    {
        a[i] = rand() % 100;
        b[i] = a[i];
    }

    // device mem copy
    int *d_a;
    CHECK(cudaMalloc((void **)&d_a, nBytes));
    CHECK(cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice));

    // num of threads
    dim3 blocks(BLOCKS, 1);   // Number of blocks
    dim3 threads(THREADS, 1); // Number of threads

    // start computation
    cudaEventRecord(start);
    int j, k;
    // external loop on comparators of size k
    for (k = 2; k <= N; k <<= 1)
    {
        // internal loop for comparator internal stages
        for (j = k >> 1; j > 0; j = j >> 1)
            bitonic_sort_step<<<blocks, threads>>>(d_a, j, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

    // recover data
    cudaMemcpy(a, d_a, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    exit(0);
}
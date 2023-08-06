#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>

#define WARP_DIMENSION 4
#define CHUNK_DIMENSION WARP_DIMENSION * 4

#define BLOCK_SIZE WARP_DIMENSION

__device__ void compare_and_swap(int* input, int k1, int k2)
{
	int elem1 = input[k1];
	int elem2 = input[k2];

	input[k1] = min(elem1, elem2);
	input[k2] = max(elem1, elem2);
}

// Returns 0 if 0, 1 if !=0
__device__ __host__ int indicator(int value)
{
	return -(value >> 31) - (-value >> 31);
}

__device__ int calculate_index(int t, int i, int j)
{
	int internTotal = (2 * i) * (2 * t / i);
	int isInsideBlock = 2 * t / i;  //indicator(2 * t / i);
	int internBlock = (2 * j) * (t / j) * isInsideBlock;
	int internInner = t % j;
	int total = internTotal + internBlock + internInner;
	return total;
}

__device__ void bitonic_last_phase(int* input, int eIdx, int localIdx, int seqLength)
{
	int k0 = eIdx + calculate_index(localIdx, seqLength, seqLength / 2);
	compare_and_swap(input, k0, k0 + seqLength / 2);

	int k1 = k0 + seqLength / 4;
	compare_and_swap(input, k1, k1 + seqLength / 2);

	for (int j = seqLength / 4; j > 0; j /= 2)
	{
		int k0 = eIdx + calculate_index(localIdx, seqLength, j);
		compare_and_swap(input, k0, k0 + j);

		int k1 = k0 + seqLength / 2;
		compare_and_swap(input, k1, k1 + j);
	}
}

__global__ void bitonic_by_a_warp(int* input, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //Global Thread Index
	int localIdx = idx % WARP_DIMENSION; //Warp Thread Index
	int divisionIdx = idx / WARP_DIMENSION; //Warp First Thread Index
	int eIdx = divisionIdx * CHUNK_DIMENSION; //Warp First Element Index

	if (idx < len / 4)
	{
		for (int i = 2; i < CHUNK_DIMENSION; i *= 2)
		{
			for (int j = i / 2; j > 0; j /= 2)
			{
				int txj = localIdx ^ j;

				int k0 = eIdx + calculate_index(localIdx, i, j);
				compare_and_swap(input, k0, k0 + j);

				int k1 = k0 + i;
				compare_and_swap(input, k1 + j, k1);
			}
		}

		// Last Phase
		bitonic_last_phase(input, eIdx, localIdx, CHUNK_DIMENSION);
	}
}

__device__ void merge_sequence(int* input, int length, int sequenceSize)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int currentSeq = length / sequenceSize;
	int currentWarp = tid / WARP_DIMENSION;
	int localTid = tid - (tid / WARP_DIMENSION * WARP_DIMENSION);

	int* currentFirstSeq = input + currentSeq * sequenceSize;
	int* currentSecondSeq = input + (currentSeq + 1) * sequenceSize;

	__shared__ int buffer[WARP_DIMENSION * 2 * (BLOCK_SIZE / WARP_DIMENSION)];

	int* bufFirst = buffer + currentWarp * WARP_DIMENSION * 2;
	int* bufSecond = bufFirst + WARP_DIMENSION;

	bufFirst[WARP_DIMENSION - localTid] = currentFirstSeq[localTid];
	bufSecond[WARP_DIMENSION - localTid] = currentSecondSeq[localTid];

	//bitonic_last_phase(input, , tid % WARP_DIMENSION, sequenceSize);
}

void warp_sort(int* input, int len)
{
	if (len % CHUNK_DIMENSION != 0)
	{
		// Error
	}


	// Step 1
	int blockSize = BLOCK_SIZE;
	int gridSize = (len / 4 + blockSize - 1) / blockSize;
	bitonic_by_a_warp << <gridSize, blockSize >> > (input, len);

	// Step 2
	for (int chunk = CHUNK_DIMENSION; chunk <= len; chunk *= 2)
	{

	}
}

__global__ void generate_input(int* data, int len, curandState* states, int max)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < len)
	{
		curand_init(tid, 0, 0, &states[tid]);
		int value = curand_uniform(&states[tid]) * max;
		data[tid] = value;
	}
}

void generate_input(int** data, int len, int max)
{
	int blockSize = BLOCK_SIZE;
	int gridSize = (len + blockSize - 1) / blockSize;
	curandState* states;
	cudaMalloc(&states, sizeof(curandState) * len);
	cudaMalloc(data, sizeof(int) * len);

	// Check malloc

	generate_input << <gridSize, blockSize >> > (*data, len, states, max);

	/*for (int i = 0; i < len; i++)
	{
		auto state = states[i];
	}*/

	// Check states
	cudaFree(states);
}

void print_array(int* arr, int len)
{
	int* copy = new int[len];
	cudaMemcpy(copy, arr, sizeof(int) * len, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	for (int i = 0; i < len; i++)
	{
		std::cout << copy[i] << " ";
	}

	std::cout << "\n";

	delete copy;
}

bool is_sorted(int* arr, int len)
{
	for (int i = 0; i < len - 1; i++)
	{
		if (arr[i] > arr[i + 1])
		{
			return false;
		}
	}
	return true;
}

int main()
{
	int len = CHUNK_DIMENSION;
	int max = len;

	int* deviceData;
	generate_input(&deviceData, len, max);

	print_array(deviceData, len);

	warp_sort(deviceData, len);

	print_array(deviceData, len);

	cudaFree(deviceData);

	getchar();

	return 0;
}

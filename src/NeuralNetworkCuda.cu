
#include <cuda_runtime.h>
#include <iostream>

#include "NeuralNetworkCuda.cuh"

NeuralNetworkCuda::NeuralNetworkCuda(int n): _n(n) {
	cudaMalloc(&d_a, n * sizeof(int));
	cudaMalloc(&d_b, n * sizeof(int));
	cudaMalloc(&d_out, n * sizeof(int));
}

NeuralNetworkCuda::~NeuralNetworkCuda() {
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
}

__global__ void addKernel(const int* a, const int* b, int* out, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] + b[i];
}

void NeuralNetworkCuda::add(const int* h_a, const int* h_b, int* h_out) {
	cudaMemcpy(d_a, h_a, _n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, _n * sizeof(int), cudaMemcpyHostToDevice);
	std::cout << "Int size: " << sizeof(int) << ".\n";

	int blockSize = 256;
	int numBlocks = (_n + blockSize - 1) / blockSize;

	addKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_out, _n);
	cudaDeviceSynchronize();

	cudaMemcpy(h_out, d_out, _n * sizeof(int), cudaMemcpyDeviceToHost);
}


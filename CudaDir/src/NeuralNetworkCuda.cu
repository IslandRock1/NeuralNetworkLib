
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

#include "NeuralNetworkCuda.cuh"

NeuralNetworkCuda::NeuralNetworkCuda(NeuralNetwork& network): network(network) {

	std::vector<float> weights;
	std::vector<float> bias;
	int numNodes = 0;

	auto layers = network.getLayers();
	for (int layerIx = 1; layerIx < layers.size(); layerIx++) {
		for (auto& n : layers[layerIx].nodes) {
			bias.emplace_back(n.bias);
			numNodes++;

			for (auto& w : n.weights) {
				weights.emplace_back(w);
			}
		}
	}

	// TODO: Size of values can be smaller, as its reused
	cudaMalloc(&values, numNodes * sizeof(float));
	cudaMalloc(&multValues, weights.size() * sizeof(float));
	cudaMalloc(&weightValues, weights.size() * sizeof(float));
	cudaMalloc(&biasValues, bias.size() * sizeof(float));

	cudaMemcpy(weightValues, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(biasValues, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);
}

NeuralNetworkCuda::~NeuralNetworkCuda() {
	cudaFree(values);
	cudaFree(multValues);
	cudaFree(weightValues);
	cudaFree(biasValues);
}

__global__ void mult(
	const float* input0,
	const float* input1,
	float* output,
	int numInput,
	int offsetInput0,
	int offsetInput1,
	int offsetOutput
	) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numInput) {
		output[i + offsetOutput] = input0[i + offsetInput0] * input1[i + offsetInput1];
	}
}

__global__ void sumArray(const float* input, float* output, int numInput, int offsetInput, int ixOutput) {
	// Can only be called using 1 thread!!
	output[ixOutput] = 0.0;
	for (int i = offsetInput; i < (offsetInput + numInput); i++) {
		output[ixOutput] += input[i];
	}
}

__global__ void LeakyReLUCuda(const float* input, float* output, int numInput, int offsetInput, int offsetOutput) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numInput) {
		auto v = input[i + offsetInput];
		output[i + offsetOutput] = fmaxf(0.01 * v, v);
	}
}

int getNumBlock(int numValues) {
	return (numValues + 255) / 256;
}

void NeuralNetworkCuda::compute(const std::vector<float>& inputValues) {

	ExecuteMult(inputValues);
	ExecuteSum();
	ExecuteActivation();
	cudaDeviceSynchronize();

	auto layers = network.getLayers();
	int numValues = layers[1].nodes.size();
	std::vector<float> out(numValues);
	cudaMemcpy(out.data(), values, numValues * sizeof(float), cudaMemcpyDeviceToHost);

	for (auto v: out) {
		std::cout << v << " ";
	}
	std::cout << "\n";
}

void NeuralNetworkCuda::ExecuteMult(const std::vector<float>& inputValues) {
	int numValues = inputValues.size();
	int blockSize = 256;
	int numBlocks = getNumBlock(numValues);

	cudaMemcpy(values, inputValues.data(), numValues * sizeof(float), cudaMemcpyHostToDevice);
	auto layers = network.getLayers();

	for (int i = 0; i < layers[1].nodes.size(); i++) {
		mult<<<numBlocks, blockSize>>>(values, weightValues, multValues, numValues, 0, i * numValues, i * numValues);
	}
}

void NeuralNetworkCuda::ExecuteSum() {
	auto layers = network.getLayers();

	int numValues = layers[1].nodes.size();
	for (int i = 0; i < numValues; i++) {
		sumArray<<<1, 1>>>(multValues, values, layers[0].nodes.size(), i * layers[0].nodes.size(), i);
	}
}

void NeuralNetworkCuda::ExecuteActivation() {
	auto layers = network.getLayers();
	int numValues = layers[1].nodes.size();
	int blockSize = 256;
	int numBlocks = getNumBlock(numValues);

	LeakyReLUCuda<<<numBlocks, blockSize>>>(values, values, numValues, 0, 0);
}

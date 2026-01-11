
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

#include "NeuralNetworkCuda.cuh"

NeuralNetworkCuda::NeuralNetworkCuda(NeuralNetwork& network)
	: network(network) {

	std::vector<float> weights;
	std::vector<float> bias;

	auto layers = network.getLayers();
	int numNodes = layers[0].nodes.size();
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

	sizeBiasValues = bias.size();
	sizeWeightValues = weights.size();
}

NeuralNetworkCuda::~NeuralNetworkCuda() {
	cudaFree(values);
	cudaFree(multValues);
	cudaFree(weightValues);
	cudaFree(biasValues);
}

void NeuralNetworkCuda::setActivationFunctions(const std::vector<int>& functions) {
	activationFunctions = functions;
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

__global__ void sumArray(const float* input, const float* bias, float* output, int numInput, int offsetInput, int offsetBias, int ixOutput) {
	// Can only be called using 1 thread!!
	// Should make better at some point
	output[ixOutput] = bias[offsetBias];
	for (int i = offsetInput; i < (offsetInput + numInput); i++) {
		output[ixOutput] += input[i];
	}
}

__global__ void LeakyReLUCuda(const float* input, float* output, int numInput, int offsetInput, int offsetOutput) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numInput) {
		auto v = input[i + offsetInput];
		output[i + offsetOutput] = fmaxf(0.01f * v, v);
	}
}

int getNumBlock(int numValues) {
	return (numValues + 255) / 256;
}

std::vector<float> NeuralNetworkCuda::compute(const std::vector<float>& inputValues) {
	sizeValues = inputValues.size();
	cudaMemcpy(values, inputValues.data(), sizeValues * sizeof(float), cudaMemcpyHostToDevice);

	auto layers = network.getLayers();
	for (int i = 1; i < layers.size(); i++) {
		ExecuteMult(i);
		ExecuteSum(i);
		ExecuteActivation(activationFunctions[i - 1], i);
	}

	cudaDeviceSynchronize();

	std::vector<float> out(sizeValues);
	cudaMemcpy(out.data(), values, sizeValues * sizeof(float), cudaMemcpyDeviceToHost);
	return out;
}

void NeuralNetworkCuda::ExecuteMult(int layerIx) {
	int numValues = sizeValues;
	int blockSize = 256;
	int numBlocks = getNumBlock(numValues);

	auto layers = network.getLayers();

	for (int i = 0; i < layers[layerIx].nodes.size(); i++) {
		mult<<<numBlocks, blockSize>>>(values, weightValues, multValues, numValues, 0, offsetWeights + i * numValues, i * numValues);
	}
	sizeMultValues = numValues * layers[layerIx].nodes.size();
	offsetWeights += layers[layerIx].nodes.size() * sizeValues;
}

void NeuralNetworkCuda::ExecuteSum(int layerIx) {
	auto layers = network.getLayers();

	int numValues = layers[layerIx].nodes.size();
	for (int i = 0; i < numValues; i++) {
		sumArray<<<1, 1>>>(multValues, biasValues, values, layers[layerIx - 1].nodes.size(), i * layers[layerIx - 1].nodes.size(), offsetBias + i, i);
	}
	sizeValues = numValues;
	offsetBias += layers[layerIx].nodes.size();
}

void NeuralNetworkCuda::ExecuteActivation(int functionIx, int layerIx) {
	auto layers = network.getLayers();
	int numValues = sizeValues;
	int blockSize = 256;
	int numBlocks = getNumBlock(numValues);

	if (functionIx == 2) {
		LeakyReLUCuda<<<numBlocks, blockSize>>>(values, values, numValues, 0, 0);
	} else if (functionIx == 5) {
		// Do nothing (linear)
	}

}

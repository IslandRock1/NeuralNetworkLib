
#include <iostream>
#include <vector>
#include <chrono>

#include "NeuralNetworkCuda.cuh"

bool isEqual(std::vector<float> v, std::vector<double> w) {
	if (v.size() != w.size()) { return false; }

	for (int i = 0; i < v.size(); i++) {
		if (std::abs(v[i] - w[i]) > 0.001) { return false; }
	}

	return true;
}

void timeAndTestSize(std::vector<int> &layerSize) {
	std::vector<int> activationFunctions;
	for (int i = 2; i < layerSize.size(); i++) {
		activationFunctions.push_back(2);
	}
	activationFunctions.push_back(5);

	std::vector<float> inputValuesF;
	std::vector<double> inputValuesD;

	for (int i = 0; i < layerSize[0]; i++) {
		inputValuesF.push_back(static_cast<float>(i));
		inputValuesD.push_back(static_cast<double>(i));
	}

	NeuralNetwork nn{layerSize};
	nn.setActivationFunctions(activationFunctions);

	NeuralNetworkCuda cuda(nn);
	cuda.setActivationFunctions(activationFunctions);

	auto t0 = std::chrono::high_resolution_clock::now();
	auto outF = cuda.compute(inputValuesF);
	auto t1 = std::chrono::high_resolution_clock::now();
	auto outD = nn.compute(inputValuesD);
	auto t2 = std::chrono::high_resolution_clock::now();

	bool same = isEqual(outF, outD);
	if (!same) {
		std::cout << "GPU: ";
		for (auto v : outF) {
			std::cout << v << " ";
		}
		std::cout << "\n";

		std::cout << "CPU: ";
		for (auto v : outD) {
			std::cout << v << " ";
		}
		std::cout << "\n";
	}
	std::cout << "Equal: " << same << "\n";

	auto timeGPU = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	auto timeCPU = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	std::cout << "GPU: " << timeGPU << " | CPU: " << timeCPU << "\n";
}

int main() {
	std::vector<int> layerSizes = {5, 2000, 10000, 5000, 200, 5};
	timeAndTestSize(layerSizes);
}

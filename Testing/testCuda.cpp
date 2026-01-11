
#include <iostream>
#include <vector>

#include "NeuralNetworkCuda.cuh"

int main() {

	std::vector<int> layerSizes = {3, 2, 1};
	NeuralNetwork nn{layerSizes};
	nn.setActivationFunctions({2, 5});

	NeuralNetworkCuda cuda(nn);
	auto outF = cuda.compute({1.0, 2.0, 3.0});
	std::cout << "Computed (GPU): ";
	for (auto v: outF) {
		std::cout << v << " ";
	}
	std::cout << "\n";

	auto outD = nn.compute({1.0, 2.0, 3.0});
	std::cout << "Computed (CPU): ";
	for (auto v : outD) {
		std::cout << v << " ";
	}
	std::cout << "\n";

	return 0;
}

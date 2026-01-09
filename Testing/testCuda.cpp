
#include <iostream>
#include <vector>

#include "NeuralNetworkCuda.cuh"

int main() {

	std::vector<int> layerSizes = {3, 2, 1};
	NeuralNetwork nn{layerSizes};

	NeuralNetworkCuda cuda(nn);
	cuda.compute({1.0, 2.0, 3.0});

	return 0;
}

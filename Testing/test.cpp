
#include <iostream>

#include "NeuralNetwork.hpp"

int main() {

	NeuralNetwork nn{{2, 100, 1000, 100, 1}};

	std::vector<double> sensorValues = {0.0, 0.0};
	nn.compute(sensorValues);

	std::cout << "Values: ";
	for (auto &v : sensorValues) {
		std::cout << " " << v;
	}
	std::cout << "\n";
}
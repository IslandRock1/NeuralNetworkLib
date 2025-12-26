
#include <iostream>

#include "Layer.hpp"

Layer::Layer(int numNodes, int numInputs, bool zeroBias) {
	for (int i = 0; i < numNodes; i++) {
		nodes.emplace_back(numInputs, zeroBias);
	}
}

std::vector<double> Layer::computeLayer(const std::vector<double>& inputValues) {
	std::vector<double> output;

	for (int i = 0; i < nodes.size(); i++) {

		double tmp = 0.0;
		auto nodeWeights = nodes[i].weights;
		for (int j = 0; j < inputValues.size(); j++) {
			tmp += inputValues[j] * nodeWeights[j];
		}

		output.push_back(nodes[i].getActivationValue(tmp));
	}

	return output;
}

void Layer::printLayer() {
	std::cout << "Layer stats: " << nodes.size() << " nodes.\n";
	for (auto &n : nodes) {
		n.printNode();
	}
}



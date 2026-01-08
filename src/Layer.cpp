
#include <iostream>
#include <Eigen/Dense>

#include "Layer.hpp"

Layer::Layer(int numNodes, int numInputs, bool zeroBias) {
	for (int i = 0; i < numNodes; i++) {
		nodes.emplace_back(numInputs, zeroBias);
	}
}

void Layer::setActivationFunction(const int functionIx) {
	for (auto& node : nodes) {
		node.setFunction(functionIx);
	}
}

std::vector<double> Layer::computeLayer(const std::vector<double>& inputValues)
{
	std::vector<double> output;
	output.reserve(nodes.size());

	// Map std::vector → Eigen (no copy)
	Eigen::Map<const Eigen::VectorXd> input(inputValues.data(),
											inputValues.size());

	for (auto& node : nodes)
	{
		Eigen::Map<const Eigen::VectorXd> weights(node.weights.data(),
												  node.weights.size());

		double tmp = input.dot(weights);   // SIMD-optimized
		output.push_back(node.getActivationValue(tmp));
	}

	return output;
}

void Layer::printLayer() {
	std::cout << "Layer stats: " << nodes.size() << " nodes.\n";
	int i = 0;
	for (auto &n : nodes) {
		std::cout << "Node: " << i++ << "/" << nodes.size() << "\n";
		n.printNode();
	}
}



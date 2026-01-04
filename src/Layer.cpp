
#include <iostream>
#include <Eigen/Dense>

#include "Layer.hpp"

Layer::Layer(int numNodes, int numInputs, bool zeroBias) {
	for (int i = 0; i < numNodes; i++) {
		nodes.emplace_back(numInputs, zeroBias);
	}
}

// std::vector<double> Layer::computeLayer(const std::vector<double>& inputValues) {
// 	std::vector<double> output;
//
// 	for (auto& node : nodes) {
//
// 		double tmp = 0.0;
// 		auto nodeWeights = node.weights;
// 		for (int j = 0; j < inputValues.size(); j++) {
// 			tmp += inputValues[j] * nodeWeights[j];
// 		}
//
// 		output.push_back(node.getActivationValue(tmp));
// 	}
//
// 	return output;
// }

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
	for (auto &n : nodes) {
		n.printNode();
	}
}



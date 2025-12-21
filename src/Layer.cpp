
#include <iostream>

#include "Layer.hpp"

Layer::Layer(int numNodes, int numInputs, bool zeroBias) {
	for (int i = 0; i < numNodes; i++) {
		_nodes.push_back({numInputs, zeroBias});
	}
}

std::vector<double> Layer::computeLayer(const std::vector<double>& inputValues) {
	std::vector<double> output;

	for (int i = 0; i < _nodes.size(); i++) {

		double tmp = 0.0;
		auto nodeWeights = _nodes[i].getWeights();
		for (int j = 0; j < inputValues.size(); j++) {
			tmp += inputValues[j] * nodeWeights[j];
		}

		output.push_back(_nodes[i].getActivationValue(tmp));
	}

	return output;
}


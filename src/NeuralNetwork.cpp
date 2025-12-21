
#include <iostream>

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<int> &layerSizes) {

	_layers.push_back({layerSizes[0], layerSizes[0], true});

	for (int i = 1; i < layerSizes.size(); i++) {
		_layers.push_back({layerSizes[i], layerSizes[i-1], false});
	}
}

std::vector<double> NeuralNetwork::compute(std::vector<double> &inputValues) {

	for (auto &layer : _layers) {
		inputValues = layer.computeLayer(inputValues);
	}

	return inputValues;
}

void NeuralNetwork::executeRandomChange(const ModificationOptions& modificationOptions) {



}



#include <iostream>
#include <fstream>
#include <random>
#include <string>

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<int> &layerSizes) {

	_numNodes = layerSizes[0];
	_layers.emplace_back(layerSizes[0], layerSizes[0], true);

	for (int i = 1; i < layerSizes.size(); i++) {
		_numNodes += layerSizes[i];
		_layers.emplace_back(layerSizes[i], layerSizes[i-1], false);
	}
}

std::vector<double> NeuralNetwork::compute(std::vector<double> &inputValues) {

	for (int i = 1; i < _layers.size(); i++) {
		inputValues = _layers[i].computeLayer(inputValues);
	}

	return inputValues;
}

void NeuralNetwork::executeRandomChange(const ModificationOptions& modificationOptions) {
	std::random_device _r;
	std::default_random_engine _e1(_r());

	std::uniform_real_distribution<double> uniform_dist_01(0, 1);
	std::uniform_real_distribution<double> uniform_dist_n11(-1, 1);

	std::uniform_int_distribution<int> uniform_dist_14(1, 4);
	std::uniform_int_distribution<int> nodePosGen(0, _numNodes - 1 - _layers[0].nodes.size());

	for (int changeNum = 0; changeNum < modificationOptions.numChanges; changeNum++) {
		int nodePos = nodePosGen(_e1);

		int layerIx = 1;
		for (int i = 1; i < _layers.size(); i++) {
			layerIx = i;

			auto nodes = _layers[i].nodes;
			if (nodePos >= nodes.size()) {
				nodePos -= nodes.size();
			} else {
				break;
			}
		}

		double changeItem = uniform_dist_01(_e1);
		if (changeItem < 0.07) {
			_layers[layerIx].nodes[nodePos].bias += uniform_dist_n11(_e1) * modificationOptions.temperature;
		} else if (changeItem < 0.10) {
			_layers[layerIx].nodes[nodePos].setFunction(uniform_dist_14(_e1));
		} else {
			std::uniform_int_distribution<int> weightPos(0, _layers[layerIx].nodes[nodePos].weights.size() - 1);
			_layers[layerIx].nodes[nodePos].weights[weightPos(_e1)] += uniform_dist_n11(_e1) * modificationOptions.temperature;
		}
	}
}

void NeuralNetwork::printNetwork() {
	std::cout << "Network stats: " << _layers.size() << " layers, " << _numNodes << " nodes.\n";
	for (auto &l : _layers) {
		l.printLayer();
	}
}

void NeuralNetwork::save(const std::string& pathS) {
	std::vector<int> sizes;
	for (auto &l : _layers) {
		sizes.push_back(l.nodes.size());
	}

	std::ofstream myFile;
	myFile.open(pathS);

	for (int i = 0; i < sizes.size() - 1; i++) {
		myFile << std::to_string(sizes[i]);
		myFile << ":";
	}
	myFile << std::to_string(sizes[sizes.size() - 1]);
	myFile << "\n";

	for (int ixL = 0; ixL < _layers.size(); ixL++) {
		for (int ixN = 0; ixN < _layers[ixL].nodes.size(); ixN++) {
			auto n = _layers[ixL].nodes[ixN];
			myFile << ixL << ":" << ixN << ":" << n.function.getRepr() << ":" << n.bias;

			for (auto &w : n.weights) {
				myFile << ":" << w;
			}
			myFile << "\n";
		}
	}

	myFile.close();
}

NeuralNetwork NeuralNetwork::copy() const {
	std::vector<int> sizes;
	for (auto &l : _layers) {
		sizes.push_back(l.nodes.size());
	}
	NeuralNetwork newNetwork{sizes};

	for (int i = 0; i < _layers.size(); i++) {
		auto& newLayer = newNetwork._layers[i];
		auto& oldLayer = _layers[i];

		for (int u = 0; u < oldLayer.nodes.size(); u++) {
			auto& newNode = newLayer.nodes[u];
			auto& oldNode = oldLayer.nodes[u];

			newNode.bias = oldNode.bias;
			newNode.function = oldNode.function;

			for (int w = 0; w < oldNode.weights.size(); w++) {
				newNode.weights[w] = oldNode.weights[w];
			}
		}
	}

	return newNetwork;
}


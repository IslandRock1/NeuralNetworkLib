
#ifndef NEURALNETWORKLIB_NEURALNETWORK_HPP
#define NEURALNETWORKLIB_NEURALNETWORK_HPP

#include <vector>

#include "ModificationOptions.hpp"
#include "Layer.hpp"

class NeuralNetwork {
public:
	NeuralNetwork(const std::vector<int> &layerSizes);
	std::vector<double> compute(std::vector<double> &inputValues);

	void executeRandomChange(const ModificationOptions &modificationOptions);
	void printNetwork();
	void save(const std::string& pathS);
	NeuralNetwork copy() const;

private:
	std::vector<Layer> _layers;
	int _numNodes;
};


#endif //NEURALNETWORKLIB_NEURALNETWORK_HPP
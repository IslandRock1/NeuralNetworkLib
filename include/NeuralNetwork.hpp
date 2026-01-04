
#ifndef NEURALNETWORKLIB_NEURALNETWORK_HPP
#define NEURALNETWORKLIB_NEURALNETWORK_HPP

#include <vector>

#include "NeuralNetwork.pb.h"

#include "ModificationOptions.hpp"
#include "Layer.hpp"

class NeuralNetwork {
public:
	explicit NeuralNetwork();
	explicit NeuralNetwork(const std::string &path);
	explicit NeuralNetwork(const std::vector<int> &layerSizes);
	std::vector<double> compute(std::vector<double> inputValues);

	void executeRandomChange(const ModificationOptions &modificationOptions);
	void printNetwork();
	void save(const std::string& pathS);
	[[nodiscard]] NeuralNetwork copy() const;

private:
	std::vector<Layer> _layers;
	int _numNodes;
};


#endif //NEURALNETWORKLIB_NEURALNETWORK_HPP
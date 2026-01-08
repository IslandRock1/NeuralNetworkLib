
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
	void setActivationFunctions(std::vector<int> activationFunctions);

	std::vector<double> compute(std::vector<double> inputValues);
	std::vector<double> softmax(std::vector<double> output);
	bool is_softmax() const;

	void executeRandomChange(const ModificationOptions &modificationOptions);
	void printNetwork();
	void save(const std::string& pathS);
	[[nodiscard]] NeuralNetwork copy() const;

private:
	std::vector<Layer> _layers;
	int _numNodes;

	bool _softmax = false;
};


#endif //NEURALNETWORKLIB_NEURALNETWORK_HPP
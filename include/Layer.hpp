
#ifndef NEURALNETWORKLIB_LAYER_HPP
#define NEURALNETWORKLIB_LAYER_HPP

#include <vector>

#include "Node.hpp"

class Layer {
public:
	Layer(int numNodes, int numInputs, bool zeroBias);
	std::vector<double> computeLayer(const std::vector<double>& inputValues);

	void printLayer();

	std::vector<Node> nodes;
};


#endif //NEURALNETWORKLIB_LAYER_HPP
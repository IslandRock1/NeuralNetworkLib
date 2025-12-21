
#ifndef NEURALNETWORKLIB_NODE_HPP
#define NEURALNETWORKLIB_NODE_HPP

#include <vector>

#include "ActivationFunctions.hpp"

class Node {
public:
	Node(int numWeights, bool zeroBias);
	double getActivationValue(double value);

	std::vector<double> getWeights();
private:

	double _bias;
	std::vector<double> _weights;
	ActivationFunctions _function;
};


#endif //NEURALNETWORKLIB_NODE_HPP
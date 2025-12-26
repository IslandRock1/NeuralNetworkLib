
#ifndef NEURALNETWORKLIB_NODE_HPP
#define NEURALNETWORKLIB_NODE_HPP

#include <vector>

#include "ActivationFunctions.hpp"

class Node {
public:
	Node(int numWeights, bool zeroBias);
	void setFunction(int funcIx);
	double getActivationValue(double value);

	void printNode();

	double bias;
	std::vector<double> weights;
	ActivationFunctions function;
};


#endif //NEURALNETWORKLIB_NODE_HPP
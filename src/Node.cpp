
#include <random>
#include <stdexcept>
#include <iostream>

#include "Node.hpp"


Node::Node(int numWeights, bool zeroBias) : function(ReLU) {
	// Random weights, bias, and function.

	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_real_distribution<double> uniform_dist(-1, 1);
	std::uniform_int_distribution<int> uniform_int(1, 4);

	if (zeroBias) {
		bias = 0.0;
	} else {
		bias = uniform_dist(e1);
	}

	for (int i = 0; i < numWeights; i++) {
		weights.emplace_back(uniform_dist(e1));
	}

	setFunction(uniform_int(e1));
}

double Node::getActivationValue(const double value) {
	return function.execute(bias + value);
}

void Node::setFunction(int funcIx) {
	switch (funcIx) {
	case 1:
		function = ActivationFunction::TanH;
		break;

	case 2:
		function = ActivationFunction::ReLU;
		break;

	case 3:
		function = ActivationFunction::Sigmoid;
		break;

	case 4:
		function = ActivationFunction::Linear;
		break;

	default:
		throw std::runtime_error("Idk..");
	}
}

int Node::getFunction() {
	if (function.getRepr() == "TanH") {return 1;}
	if (function.getRepr() == "ReLU") {return 2;}
	if (function.getRepr() == "Sigmoid") {return 3;}
	if (function.getRepr() == "Linear") {return 4;}

	return -1;
}


void Node::printNode() {
	std::cout << "Bias: " << bias << " | Function: " << function.getRepr() << "\n";
	std::cout << "Weights:";
	for (auto &w : weights) {
		std::cout << " " << w;
	}
	std::cout << "\n";
}


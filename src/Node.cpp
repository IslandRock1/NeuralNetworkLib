
#include <random>

#include "Node.hpp"

#include <stdexcept>

Node::Node(int numWeights, bool zeroBias) : _function(ReLU) {
	// Random weights, bias, and function.

	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_real_distribution<double> uniform_dist(-1, 1);
	std::uniform_int_distribution<int> uniform_int(1, 4);

	if (zeroBias) {
		_bias = 0.0;
	} else {
		_bias = uniform_dist(e1);
	}

	for (int i = 0; i < numWeights; i++) {
		_weights.emplace_back(uniform_dist(e1));
	}

	switch (uniform_int(e1)) {
	case 1:
		_function = ActivationFunction::TanH;
		break;

	case 2:
		_function = ActivationFunction::ReLU;
		break;

	case 3:
		_function = ActivationFunction::Sigmoid;
		break;

	case 4:
		_function = ActivationFunction::Linear;
		break;

	default:
		throw std::runtime_error("Idk..");
	}
}

double Node::getActivationValue(const double value) {
	return _function.execute(_bias + value);
}

std::vector<double> Node::getWeights() {
	return _weights;
}


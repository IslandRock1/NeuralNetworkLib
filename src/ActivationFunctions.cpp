
#include <cmath>
#include "ActivationFunctions.hpp"

ActivationFunctions::ActivationFunctions(const ActivationFunction function): _function(function) {}

double ActivationFunctions::execute(const double value) {
	switch (_function) {
		case ActivationFunction::TanH: {
			return TanH(value);
		}

		case ActivationFunction::ReLU: {
			return ReLU(value);
		}

		case ActivationFunction::Sigmoid: {
			return Sigmoid(value);
		}

		case ActivationFunction::Linear: {
			return Linear(value);
		}
	}
}

double ActivationFunctions::TanH(const double value) {
	return std::tanh(value);
}

double ActivationFunctions::ReLU(const double value) {
	return value ? value > 0 : 0.0;
}

double ActivationFunctions::Sigmoid(const double value) {
	if (value < -200.0) {return 0.0;}
	if (value >   40.0) {return 0.0;}

	return 1 / (1 + std::exp(-value));
}

double ActivationFunctions::Linear(const double value) {
	return value;
}


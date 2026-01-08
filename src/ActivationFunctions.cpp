
#include <cmath>
#include <iostream>
#include "ActivationFunctions.hpp"

ActivationFunctions::ActivationFunctions(const ActivationFunction function): _function(function) {}

double ActivationFunctions::execute(const double value) const {
	switch (_function) {
		case ActivationFunction::TanH: {
			return TanH(value);
		}

		case ActivationFunction::LeakyReLU: {
			return LeakyReLU(value);
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

		default: {
			throw std::runtime_error("Invalid function.");
		}
	}
}

double ActivationFunctions::TanH(const double value) {
	return std::tanh(value);
}

double ActivationFunctions::LeakyReLU(double value) {
	return std::max(0.01 * value, value);
}


double ActivationFunctions::ReLU(const double value) {
	return value > 0 ? value : 0.0;
}

double ActivationFunctions::Sigmoid(const double value) {
	if (value < -200.0) {return 0.0;}
	if (value >   40.0) {return 0.0;}

	return 1 / (1 + std::exp(-value));
}

double ActivationFunctions::Linear(double value) {
	return value;
}


std::string ActivationFunctions::getRepr() const {
	switch (_function) {
		case ActivationFunction::TanH: {
			return "TanH";
		}

		case ActivationFunction::LeakyReLU: {
			return "LeakyReLU";
		}

		case ActivationFunction::ReLU: {
			return "ReLU";
		}

		case ActivationFunction::Sigmoid: {
			return "Sigmoid";
		}

		case ActivationFunction::Linear: {
				return "Linear";
		}

		default: {
			throw std::runtime_error("Invalid function.");
		}
	}
}



#ifndef NEURALNETWORKLIB_ACTIVATIONFUNCTIONS_HPP
#define NEURALNETWORKLIB_ACTIVATIONFUNCTIONS_HPP

enum ActivationFunction {
	TanH,
	ReLU,
	Sigmoid,
	Linear
};

class ActivationFunctions {
public:
	ActivationFunctions(ActivationFunction function);
	double execute(double value);
	std::string getRepr();

private:
	ActivationFunction _function;
	static double TanH(double value);
	static double ReLU(double value);
	static double Sigmoid(double value);
	static double Linear(double value);
};

#endif //NEURALNETWORKLIB_ACTIVATIONFUNCTIONS_HPP
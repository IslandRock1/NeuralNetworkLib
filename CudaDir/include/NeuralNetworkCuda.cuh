
#ifndef NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH
#define NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH

#include "NeuralNetwork.hpp"

class NeuralNetworkCuda {
public:
	NeuralNetworkCuda(NeuralNetwork& network);
	~NeuralNetworkCuda();

	void compute(const std::vector<float>& inputValues);
private:
	NeuralNetwork& network;

	float* values;
	float* multValues;
	float* biasValues;
	float* weightValues;

	void ExecuteMult(const std::vector<float>& inputValues);
	void ExecuteSum();
	void ExecuteActivation();
};


#endif //NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH
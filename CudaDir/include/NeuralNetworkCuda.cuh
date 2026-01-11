
#ifndef NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH
#define NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH

#include "NeuralNetwork.hpp"

class NeuralNetworkCuda {
public:
	NeuralNetworkCuda(NeuralNetwork& network);
	~NeuralNetworkCuda();

	void setActivationFunctions(const std::vector<int>& functions);

	std::vector<float> compute(const std::vector<float>& inputValues);
private:
	NeuralNetwork& network;
	std::vector<int> activationFunctions;

	float* values;
	float* multValues;
	float* biasValues;
	float* weightValues;

	int sizeValues = 0;
	int sizeMultValues = 0;

	int offsetBias = 0;
	int offsetWeights = 0;

	void ExecuteMult(int layerIx);
	void ExecuteSum(int layerIx);
	void ExecuteActivation(int functionIx, int layerIx);
};


#endif //NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH
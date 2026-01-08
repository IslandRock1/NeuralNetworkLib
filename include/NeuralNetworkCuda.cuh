
#ifndef NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH
#define NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH

class NeuralNetworkCuda {
public:
	NeuralNetworkCuda(int n);
	~NeuralNetworkCuda();

	void add(const int* h_a, const int* h_b, int* h_out);
private:
	int _n;
	int* d_a = nullptr;
	int* d_b = nullptr;
	int* d_out = nullptr;
};


#endif //NEURALNETWORKLIB_NEURALNETWORKCUDA_CUH
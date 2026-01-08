
#include <iostream>
#include <vector>

#include "NeuralNetworkCuda.cuh"

int main() {
	const int N = 1000;

	std::vector<int> a(N), b(N), out(N);

	for (int i = 0; i < N; ++i)
	{
		a[i] = int(i);
		b[i] = int(i * 2);
	}

	NeuralNetworkCuda cuda(N);
	cuda.add(a.data(), b.data(), out.data());

	for (int i = 0; i < N; ++i)
		std::cout << out[i] << " ";

	std::cout << "\n";
	return 0;
}

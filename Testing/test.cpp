
#include <iostream>

#include "NeuralNetwork.hpp"
#include "ModificationOptions.hpp"

int main() {

	NeuralNetwork nn{{2, 1}};

	nn.printNetwork();

	for (int i = 0; i < 100; i++) {
		ModificationOptions modification_options{1.0};
		nn.executeRandomChange(modification_options);
	}
	ModificationOptions modification_options{1.0};
	nn.executeRandomChange(modification_options);

	std::cout << "##########################\n";
	nn.printNetwork();
}
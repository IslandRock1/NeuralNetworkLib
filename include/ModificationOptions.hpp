
#ifndef NEURALNETWORKLIB_MODIFICATIONOPTIONS_HPP
#define NEURALNETWORKLIB_MODIFICATIONOPTIONS_HPP

struct ModificationOptions {
	int numChanges;
	double temperature; // scale
	double chanceChangeFunction;
	double chanceChangeBias;
	double chanceChangeWeights;
};

#endif //NEURALNETWORKLIB_MODIFICATIONOPTIONS_HPP
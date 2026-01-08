
#ifndef NEURALNETWORKLIB_COMUNICATIONSTRUCT_HPP
#define NEURALNETWORKLIB_COMUNICATIONSTRUCT_HPP

#include <string>
#include <vector>

struct PythonToCPP {
	std::string projectName;

	std::vector<int> hiddenLayerSize; // In and out is decided by sim
	std::vector<int> activationFunctions;
	int networksPerIter;
	int numSimulations;
	double simTime;

	double percentNetworksKept;
	double percentNetworksNew;
	double percentNetworksModifiable;

	double percentChangeFunction;
	double percentChangeBias;
	double percentChangeWeight;

	int numChanges;
	double temperature;
	std::vector<double> constValues;
};

struct CPPToPython {
	std::vector<double> bestScores = {};
	std::vector<double> avgScores = {};
	std::vector<double> validationScores = {};

	int finishedIterations = 0;
	int finishedNetworksThisIter = 0;

	double timePerMutate = 0.0;
	double timePerNetwork = 0.0;
	double timePerIter = 0.0;
	double timeComputationNetwork = 0.0;
	double timeComputetionSimulation = 0.0;
};

#endif //NEURALNETWORKLIB_COMUNICATIONSTRUCT_HPP
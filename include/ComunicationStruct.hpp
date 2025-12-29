
#ifndef NEURALNETWORKLIB_COMUNICATIONSTRUCT_HPP
#define NEURALNETWORKLIB_COMUNICATIONSTRUCT_HPP

#include <string>
#include <vector>

struct PythonToCPP {
	std::string projectName;

	std::vector<int> hiddenLayerSize; // In and out is decided by sim
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
	std::vector<double> bestScores;
	std::vector<double> avgScores;

	int finishedIterations;
	int finishedNetworksThisIter;

	double timePerMutate;
	double timePerNetwork;
	double timePerIter;
	double timeComputationNetwork;
	double timeComputetionSimulation;
};

#endif //NEURALNETWORKLIB_COMUNICATIONSTRUCT_HPP
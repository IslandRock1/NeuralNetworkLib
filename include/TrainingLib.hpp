
#ifndef NEURALNETWORKLIB_TRAININGLIB_HPP
#define NEURALNETWORKLIB_TRAININGLIB_HPP

#include <mutex>
#include <thread>
#include <vector>
#include <atomic>

#include "NeuralNetwork.hpp"
#include "ComunicationStruct.hpp"

class TrainingLib {
public:
	TrainingLib(PythonToCPP settings);

	void stop();
	void setInfo(PythonToCPP &settings);
	CPPToPython getInfo();

private:
	void _run();
	void _step();
	void _testIteration();
	void _threadTask();
	void _mutation();
	double _testSimulation(NeuralNetwork &network);

	void mutationTask();

	PythonToCPP _settings;
	CPPToPython _outputStats;
	std::mutex _pythonMutex;

	int _iteration = 0;
	std::vector<int> _layerSizes;

	std::vector<double> _threadScores;
	std::vector<std::pair<double, NeuralNetwork>> _scores;

	std::atomic_bool _stopFlag = false;
	std::jthread _computeOrganizingThread;
	std::vector<NeuralNetwork> _networks;
	std::vector<double> _bestScores;
	std::vector<double> _avgScores;

	std::chrono::duration<double> _totalTimePerMutation{};
	long long unsigned int _totalMutationsDone = 0;
	std::mutex _mutationMutex;

	std::chrono::duration<double> _totalTimePerNetwork{};
	long long unsigned int _totalNetworksTested = 0;
	std::mutex _networkMutex;

	std::chrono::duration<double> _totalTimePerIteration{};
	long long unsigned int _totalIterationsTested = 0;
	std::mutex _iterationMutex;

	std::chrono::duration<double> _totalTimeCompute{};
	std::chrono::duration<double> _totalTimeSim{};
	std::mutex _computeSimMutex;

	std::atomic<size_t> nextIndex{0};
	std::atomic<bool> stopThreads{false};

	const size_t numThreads = std::thread::hardware_concurrency() - 1;
	std::vector<std::thread> workers;

	std::vector<std::thread> mutationWorkers;
	std::vector<NeuralNetwork> newNetworks;

	std::atomic<size_t> nextIndexMutation{0};
	std::atomic<bool> stopThreadsMutation{false};
	int numKeepNetworks = 0;
	int numRandomNetworks = 0;
	int numModifiedNetworks = 0;
	int numModifiableNetworks = 0;
};


#endif //NEURALNETWORKLIB_TRAININGLIB_HPP
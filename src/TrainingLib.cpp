
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>
#include <filesystem>

#include "TrainingLib.hpp"
#include "Tester.hpp"

TrainingLib::TrainingLib(PythonToCPP &settings)
	: _settings(std::move(settings)) {

	_outputStats = CPPToPython{{}, {}, 0, 0, 0.0, 0.0, 0.0, 0.0};

	_layerSizes = {Tester::numInputNodes};
	for (auto &lSize : _settings.hiddenLayerSize) {
		_layerSizes.emplace_back(lSize);
	}
	_layerSizes.emplace_back(Tester::numOutputNodes);

	while (_networks.size() < _settings.networksPerIter) {
		NeuralNetwork nn{_layerSizes};
		_networks.emplace_back(nn);
	}

	_computeOrganizingThread = std::jthread{&TrainingLib::_run, this};

	std::filesystem::create_directory("models");
}

void TrainingLib::stop() {
	std::cout << "Stopping..\n";
	_stopFlag = true;

	if (_computeOrganizingThread.joinable()) {
		_computeOrganizingThread.join();
	}

	std::cout << "Stopped all threads..?\n";
}

void TrainingLib::setInfo(PythonToCPP& settings) {
	std::lock_guard<std::mutex> lock(_pythonMutex);
	_settings = settings;
}

CPPToPython TrainingLib::getInfo() {
	double timePerMutation = -1;
	if (_totalMutationsDone > 0) {
		std::lock_guard lock(_mutationMutex);
		timePerMutation = std::chrono::duration_cast<std::chrono::milliseconds>(_totalTimePerMutation).count() / _totalMutationsDone;
	}

	double timePerIteration = -1;
	if (_totalIterationsTested > 0) {
		std::lock_guard lock(_networkMutex);
		timePerIteration = std::chrono::duration_cast<std::chrono::milliseconds>(_totalTimePerIteration).count() / _totalIterationsTested;
	}

	double timePerNetwork = -1;
	if (_totalNetworksTested > 0) {
		std::lock_guard lock(_networkMutex);
		timePerNetwork = std::chrono::duration_cast<std::chrono::milliseconds>(_totalTimePerNetwork).count() / _totalNetworksTested;
	}

	double totalSimTime;
	double totalComputeTime;
	{
		std::lock_guard lock(_computeSimMutex);
		totalSimTime = std::chrono::duration_cast<std::chrono::milliseconds>(_totalTimeSim).count();
		totalComputeTime = std::chrono::duration_cast<std::chrono::milliseconds>(_totalTimeCompute).count();
	}

	std::lock_guard<std::mutex> lock(_pythonMutex);
	_outputStats.timePerMutate = timePerMutation;
	_outputStats.timePerNetwork = timePerNetwork;
	_outputStats.timePerIter = timePerIteration;
	_outputStats.timeComputationNetwork = totalComputeTime;
	_outputStats.timeComputetionSimulation = totalSimTime;

	return _outputStats;
}

void TrainingLib::_run() {
	while (!_stopFlag) {
		_step();
	}
}

bool sortPred(std::pair<double, NeuralNetwork> &left, std::pair<double, NeuralNetwork> &right) {
	return left.first > right.first;
}

void TrainingLib::_step() {
	_iteration++;

	const auto t0 = std::chrono::high_resolution_clock::now();
	_testIteration(); // Updates scores
	const auto t1 = std::chrono::high_resolution_clock::now();

	if (_stopFlag) {return;}

	{
		std::lock_guard<std::mutex> lock(_iterationMutex);
		_totalTimePerIteration += (t1 - t0);
		_totalIterationsTested++;
	}

	double bestScore = 0.0;
	double tot = 0.0;
	double num = 0.0;

	std::ranges::sort(_scores, sortPred);

	std::string path = "models/" + std::to_string(_iteration);
	_scores[0].second.save(path);
	bestScore = _scores[0].first;
	for (auto &s : _scores) {
		tot += s.first;
		num += 1;
	}

	{
		std::lock_guard<std::mutex> lock(_pythonMutex);
		_outputStats.bestScores.emplace_back(bestScore);
		_outputStats.avgScores.emplace_back(tot / num);
	}

	auto t0_mut = std::chrono::high_resolution_clock::now();
	_mutation();
	auto t1_mut = std::chrono::high_resolution_clock::now();

	{
		std::lock_guard<std::mutex> lock(_mutationMutex);
		_totalTimePerMutation += (t1_mut - t0_mut);
		_totalMutationsDone++;
	}
}

void TrainingLib::_testIteration() {
	{
		std::lock_guard<std::mutex> lock(_pythonMutex);
		_outputStats.finishedNetworksThisIter = 0;
	}

	_scores.clear();

	std::chrono::duration<double> networkTime{};
	int numNetworks = 0;

	for (auto &nn : _networks) {
		double totReward = 0.0;

		auto t0 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < _settings.numSimulations; i++) {
			totReward += _testSimulation(nn);
			if (_stopFlag) {return;}
		}
		auto t1 = std::chrono::high_resolution_clock::now();
		networkTime += (t1 - t0);
		numNetworks++;

		_scores.emplace_back(totReward, nn.copy());
		if (_stopFlag) {return;}

		{
			std::lock_guard<std::mutex> lock(_pythonMutex);
			_outputStats.finishedNetworksThisIter++;
		}
	}

	{
		std::lock_guard<std::mutex> lock(_pythonMutex);
		_outputStats.finishedIterations++;
	}

	{
		std::lock_guard<std::mutex> lock(_networkMutex);
		_totalTimePerNetwork += networkTime;
		_totalNetworksTested += numNetworks;
	}
}

double TrainingLib::_testSimulation(NeuralNetwork& network) {
	Tester tester{};

	std::chrono::duration<double> computeTime{};
	std::chrono::duration<double> simTime{};

	while (true) {
		auto networkInfo = tester.getInfo();

		auto t0 = std::chrono::high_resolution_clock::now();
		auto output = network.compute(networkInfo);
		auto t1 = std::chrono::high_resolution_clock::now();
		auto simInfo = tester.update(output[0], _settings.constValues);
		auto t2 = std::chrono::high_resolution_clock::now();
		if (_stopFlag) {return 0.0;}

		computeTime += (t1 - t0);
		simTime += (t2 - t1);

		if (simInfo.isFinished) {

			{
				std::lock_guard<std::mutex> lock(_computeSimMutex);
				_totalTimeCompute += computeTime;
				_totalTimeSim += simTime;
			}

			return simInfo.reward;
		}
	}
}

double randomDouble(double min = 0.0, double max = 1.0) {
	static std::random_device rd;
	static std::mt19937 gen(rd());

	std::uniform_real_distribution<double> dist(min, max);
	return dist(gen);
}

int randomInt(int min = 0.0, int max = 1.0) {
	static std::random_device rd;   // Non-deterministic seed
	static std::mt19937 gen(rd());  // Mersenne Twister RNG

	std::uniform_int_distribution<int> dist(min, max);
	return dist(gen);
}

void TrainingLib::_mutation() {
	// TMP!

	int numKeptNetworks = static_cast<int>(_settings.percentNetworksKept * _settings.networksPerIter);
	int numNewNetworks = static_cast<int>(_settings.percentNetworksNew * _settings.networksPerIter);
	int numModifiableNetworks = static_cast<int>(_settings.percentNetworksModifiable * _settings.networksPerIter);

	std::vector<NeuralNetwork> newNetworks;
	for (int i = 0; i < numKeptNetworks; i++) {
		auto net = _scores[i].second;
		newNetworks.emplace_back(net.copy());

		if (_stopFlag) {return;}
	}

	for (int i = 0; i < numNewNetworks; i++) {
		NeuralNetwork nn{_layerSizes};
		newNetworks.emplace_back(nn);

		if (_stopFlag) {return;}
	}

	while (newNetworks.size() < _settings.networksPerIter) {
		auto randIx = randomInt(0, numModifiableNetworks);
		auto networkCopy = _scores[randIx].second.copy();

		networkCopy.executeRandomChange({
			_settings.numChanges,
			_settings.temperature,
			_settings.percentChangeFunction,
			_settings.percentChangeBias,
			_settings.percentChangeWeight});

		newNetworks.emplace_back(networkCopy);

		if (_stopFlag) {return;}
	}

	_networks = std::move(newNetworks);
}



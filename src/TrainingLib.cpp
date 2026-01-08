
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>
#include <filesystem>

#include "TrainingLib.hpp"
#include "PendulumTester.hpp"
#include "DoublePendulumTester.hpp"
#include "MNISTDigitTester.hpp"

TrainingLib::TrainingLib(PythonToCPP settings)
	: _settings(std::move(settings)), _threadScores(_settings.networksPerIter) {

	_outputStats = CPPToPython{};

	if (_settings.projectName == "Pendulum") {
		testerFactory = [] {
			return std::make_unique<Pendulum::PendulumTester>();
		};
	} else if (_settings.projectName == "DoublePendulum") {
		testerFactory = [] {
			return std::make_unique<DoublePendulum::DoublePendulumTester>();
		};
	} else if (_settings.projectName == "MNIST") {
		testerFactory = [] {
			return std::make_unique<MNISTDigit::MNISTDigitTester>();
		};
	} else {
		stop();
		throw std::runtime_error("Invalid project name.");
	}

	auto tmpTester = testerFactory();
	int input = tmpTester->numInputNodes();
	int output = tmpTester->numOutputNodes();

	_layerSizes = {input};
	for (auto &lSize : _settings.hiddenLayerSize) {
		_layerSizes.emplace_back(lSize);
	}
	_layerSizes.emplace_back(output);

	while (_networks.size() < _settings.networksPerIter) {
		NeuralNetwork nn{_layerSizes};

		if (! _settings.activationFunctions.empty()) {
			nn.setActivationFunctions(_settings.activationFunctions);
		}

		_networks.emplace_back(nn);
	}

	_computeOrganizingThread = std::jthread{&TrainingLib::_run, this};

	auto path = std::filesystem::current_path();
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
		timePerMutation = std::chrono::duration_cast<std::chrono::microseconds>(_totalTimePerMutation).count() / _totalMutationsDone;
	}

	double timePerIteration = -1;
	if (_totalIterationsTested > 0) {
		std::lock_guard lock(_networkMutex);
		timePerIteration = std::chrono::duration_cast<std::chrono::microseconds>(_totalTimePerIteration).count() / _totalIterationsTested;
	}

	double timePerNetwork = -1;
	if (_totalNetworksTested > 0) {
		std::lock_guard lock(_networkMutex);
		timePerNetwork = std::chrono::duration_cast<std::chrono::microseconds>(_totalTimePerNetwork).count() / _totalNetworksTested;
	}

	double totalSimTime;
	double totalComputeTime;
	{
		std::lock_guard lock(_computeSimMutex);
		totalSimTime = std::chrono::duration_cast<std::chrono::microseconds>(_totalTimeSim).count();
		totalComputeTime = std::chrono::duration_cast<std::chrono::microseconds>(_totalTimeCompute).count();
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

	auto t0_mut = std::chrono::high_resolution_clock::now();
	_mutation();
	auto t1_mut = std::chrono::high_resolution_clock::now();

	{
		std::lock_guard<std::mutex> lock(_mutationMutex);
		_totalTimePerMutation += (t1_mut - t0_mut);
		_totalMutationsDone++;
	}

	auto tester = testerFactory();
	tester->setValidation();
	auto network = _networks[0];
	double score = 0.0;

	while (true) {
		auto networkInfo = tester->getInfo();

		auto output = network.compute(networkInfo);
		if (network.is_softmax()) {output = network.softmax(output);}
		auto simInfo = tester->update(output, _settings.constValues);

		if (simInfo.isFinished) {
			score = simInfo.reward;
			break;
		}
	}

	{
		std::lock_guard<std::mutex> lock(_pythonMutex);
		_outputStats.bestScores.emplace_back(bestScore);
		_outputStats.avgScores.emplace_back(tot / num);
		_outputStats.validationScores.emplace_back(score);
	}
}

void TrainingLib::_testIteration() {
	{
		std::lock_guard<std::mutex> lock(_pythonMutex);
		_outputStats.finishedNetworksThisIter = 0;
	}

	nextIndex.store(0, std::memory_order_relaxed);
	stopThreads.store(false);

	_threadScores.clear();
	_threadScores.resize(_networks.size());
	_scores.clear();

	std::chrono::duration<double> networkTime{};
	int numNetworks = 0;

	for (int i = 0; i < numThreads; i++) {
		workers.emplace_back(&TrainingLib::_threadTask, this);
	}

	for (auto& t : workers) {
		t.join();
	}

	workers.clear();

	// After all threads are finished
	for (int ix = 0; ix < _networks.size(); ix++) {
		_scores.emplace_back(_threadScores[ix], _networks[ix].copy());
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

void TrainingLib::_threadTask() {

	while (true) {
		size_t ix = nextIndex.fetch_add(1, std::memory_order_relaxed);
		if (ix >= _networks.size() || stopThreads.load()) return;

		double totReward = 0.0;
		NeuralNetwork &network = _networks[ix];

		auto t0 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < _settings.numSimulations; i++) {
			totReward += _testSimulation(network);
			if (_stopFlag) {return;}
		}
		auto t1 = std::chrono::high_resolution_clock::now();

		{
			std::lock_guard<std::mutex> lock(_networkMutex);
			_totalTimePerNetwork += (t1 - t0);
			_totalNetworksTested++;
		}

		_threadScores[ix] = totReward;
		if (_stopFlag) {return;}

		{
			std::lock_guard<std::mutex> lock(_pythonMutex);
			_outputStats.finishedNetworksThisIter++;
		}
	}
}

double TrainingLib::_testSimulation(NeuralNetwork& network) {
	auto tester = testerFactory();

	std::chrono::duration<double> computeTime{};
	std::chrono::duration<double> simTime{};

	while (true) {
		auto networkInfo = tester->getInfo();

		auto t0 = std::chrono::high_resolution_clock::now();
		auto output = network.compute(networkInfo);
		if (network.is_softmax()) {
			output = network.softmax(output);
		}

		auto t1 = std::chrono::high_resolution_clock::now();
		auto simInfo = tester->update(output, _settings.constValues);
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
	numKeepNetworks = static_cast<int>(_settings.percentNetworksKept * _settings.networksPerIter);
	numRandomNetworks = static_cast<int>(_settings.percentNetworksNew * _settings.networksPerIter);
	numModifiableNetworks = static_cast<int>(_settings.percentNetworksModifiable * _settings.networksPerIter);
	numModifiedNetworks = _settings.networksPerIter - numKeepNetworks - numRandomNetworks;

	newNetworks.clear();
	newNetworks.resize(_settings.networksPerIter, _networks[0].copy());

	nextIndexMutation.store(0, std::memory_order_relaxed);
	stopThreadsMutation.store(false);

	for (int i = 0; i < numThreads; i++) {
		mutationWorkers.emplace_back(&TrainingLib::mutationTask, this);
	}

	for (auto& t : mutationWorkers) {
		t.join();
	}

	mutationWorkers.clear();

	_networks = std::move(newNetworks);
}

void TrainingLib::mutationTask() {
	while (true) {
		size_t ix = nextIndexMutation.fetch_add(1, std::memory_order_relaxed);
		if (ix >= _settings.networksPerIter || stopThreadsMutation.load()) return;

		if (ix < numKeepNetworks) {
			auto net = _scores[ix].second;
			newNetworks[ix] = net.copy();
		} else if (ix < numKeepNetworks + numRandomNetworks) {
			NeuralNetwork nn{_layerSizes};

			if (! _settings.activationFunctions.empty()) {
				nn.setActivationFunctions(_settings.activationFunctions);
			}

			newNetworks[ix] = nn;
		} else if (ix < _settings.networksPerIter) {
			auto randIx = randomInt(0, numModifiableNetworks);
			auto networkCopy = _scores[randIx].second.copy();

			networkCopy.executeRandomChange({
				_settings.numChanges,
				_settings.temperature,
				_settings.percentChangeFunction,
				_settings.percentChangeBias,
				_settings.percentChangeWeight});

			if (! _settings.activationFunctions.empty()) {
				networkCopy.setActivationFunctions(_settings.activationFunctions);
			}

			newNetworks[ix] = networkCopy;
		}

	}
}




#ifndef NEURALNETWORKLIB_TESTER_HPP
#define NEURALNETWORKLIB_TESTER_HPP

#include <vector>

#include "PendulumSimulation.hpp"

struct SimulationInfo {
	bool isFinished;
	double progress;
	double reward;
};

class Tester {
public:

	Tester();
	Tester(const PendulumInfo &pendulum);
	std::vector<double> getInfo() const;
	SimulationInfo update(double externalForce, const std::vector<double> &constValues);

	static constexpr int numInputNodes = 5;
	static constexpr int numOutputNodes = 1;

private:
	PendulumSimulation _simulation;
	double _dt = 1.0 / 60.0;
	double _maxForce = 1.0;

	double _sumTime = 0.0;
	double _maxTime = 20.0;

	int _runningCount = 0;
	double _sumReward = 0.0;

	double rewardFunction(const std::vector<double> &constValues);
};


#endif //NEURALNETWORKLIB_TESTER_HPP
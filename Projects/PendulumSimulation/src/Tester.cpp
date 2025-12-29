
#include <iostream>
#include <cmath>

#include "Tester.hpp"

Tester::Tester() = default;

Tester::Tester(const PendulumInfo& pendulum): _simulation(pendulum) {}

std::vector<double> Tester::getInfo() const {
	// pendulumX, pendulumY, rotationVel, cartPos, cartVel

	auto [anglePos, angleVel, cartPos, cartVel] = _simulation.getInfo();
	const double pendulumX = std::sin(anglePos);
	const double pendulumY = std::cos(anglePos);
	return {pendulumX, pendulumY, angleVel, cartPos, cartVel};
}

SimulationInfo Tester::update(const double externalForce, const std::vector<double>& constValues) {
	const auto force = std::max(-_maxForce, std::min(_maxForce, externalForce));

	_sumTime += _dt;
	_simulation.update(_dt, force);

	const bool isFinished = _sumTime >= _maxTime;
	const double progress = _sumTime / _maxTime;
	_sumReward += rewardFunction(constValues);

	return {isFinished, progress, _sumReward};
}

double Tester::rewardFunction(const std::vector<double>& constValues) {
	auto [anglePos, angleVel, cartPos, cartVel] = _simulation.getInfo();
	const double error = abs(anglePos);
	const bool overThreshold = (error <= (3.1415926 / 3.0));
	_runningCount = (_runningCount + 1) * overThreshold;

	double angleVelTerm = angleVel * angleVel;
	double cartPosTerm = cartPos * cartPos;
	double cartVelTerm = cartVel * cartVel;

	if (constValues.size() >= 4) {
		return constValues[0] * _runningCount - constValues[1] * angleVelTerm - constValues[2] * cartPosTerm - constValues[3] * cartVelTerm;
	}
	return _runningCount; // - angleVelTerm - cartPosTerm - cartVelTerm;
}



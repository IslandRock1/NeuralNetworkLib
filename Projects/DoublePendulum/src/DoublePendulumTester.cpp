
#include <cmath>
#include "DoublePendulumTester.hpp"

DoublePendulum::DoublePendulumTester::DoublePendulumTester() = default;

std::vector<double> DoublePendulum::DoublePendulumTester::getInfo() {
	// pendulumX0, pendulumX1, pendulumY0, pendulumY1,
	// rotationVel0, rotationVel1, cartPos, cartVel

	auto v = _simulation.getInfo();

	double pendulumX0 = std::sin(v.anglePos0);
	double pendulumX1 = std::sin(v.anglePos1);

	double pendulumY0 = std::cos(v.anglePos0);
	double pendulumY1 = std::cos(v.anglePos1);

	return {pendulumX0, pendulumX1, pendulumY0, pendulumY1,
		v.angleVel0, v.angleVel1, v.cartPos, v.cartVel};
}

SimulationInfo DoublePendulum::DoublePendulumTester::update(std::vector<double>& network, std::vector<double>& rewardModifiers) {
	const auto force = std::max(-_maxForce, std::min(_maxForce, network[0]));

	_sumTime += _dt;
	_simulation.update(_dt, force * 10.0);

	const bool isFinished = _sumTime >= _maxTime;
	const double progress = _sumTime / _maxTime;
	_sumReward += rewardFunction(rewardModifiers);

	return {isFinished, progress, _sumReward};
}

double DoublePendulum::DoublePendulumTester::rewardFunction(const std::vector<double>& constValues) {
	auto [anglePos0, anglePos1, angleVel0, angleVel1, cartPos, cartVel] = _simulation.getInfo();
	const double error0 = abs(anglePos0);
	const double error1 = abs(anglePos1);

	double div = 5.0;
	if (constValues.size() >= 5) {div = constValues[4];}
	const bool overThreshold0 = (error0 <= (3.1415926 / div));
	const bool overThreshold1 = (error1 <= (3.1415926 / div));
	_runningCount = (_runningCount + 1) * overThreshold0 * overThreshold1;

	double angleVelTerm = angleVel0 * angleVel0 + angleVel1 * angleVel1;
	double cartPosTerm = cartPos * cartPos;
	double cartVelTerm = cartVel * cartVel;

	if (constValues.size() >= 4) {
		return constValues[0] * _runningCount - constValues[1] * angleVelTerm - constValues[2] * cartPosTerm - constValues[3] * cartVelTerm;
	}
	return _runningCount;
}



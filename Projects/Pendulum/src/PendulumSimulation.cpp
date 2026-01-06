
#include <cmath>
#include <random>

#include "PendulumSimulation.hpp"

double randomPendulum(double min = 0.0, double max = 1.0) {
	static std::random_device rd;
	static std::mt19937 gen(rd());

	std::uniform_real_distribution<double> dist(min, max);
	return dist(gen);
}

PendulumSimulation::PendulumSimulation() {
	double offset = 0.01;
	double pi = 3.1415926;
	_anglePos = randomPendulum(pi-offset, pi+offset);
	_angleVel = randomPendulum(-0.1, 0.1);

	_cartPos = randomPendulum(-0.1, 0.1);
	_cartVel = randomPendulum(-0.1, 0.1);
}

PendulumSimulation::PendulumSimulation(const PendulumInfo& info)
	: _anglePos(info.anglePos), _angleVel(info.angleVel),
	_cartPos(info.cartPos), _cartVel(info.cartVel) {}

PendulumSimulation::PendulumSimulation(const double anglePos, const double angleVel, const double cartPos, const double cartVel)
	: _anglePos(anglePos), _angleVel(angleVel),
	_cartPos(cartPos), _cartVel(cartVel) {}

PendulumInfo PendulumSimulation::getInfo() const {
	return {_anglePos, _angleVel, _cartPos, _cartVel};
}

void PendulumSimulation::update(const double dt, const double externalForce) {
	double force;
	if (externalForce > _maxForce) {
		force = _maxForce;
	} else if (externalForce < -_maxForce) {
		force = -_maxForce;
	} else {
		force = externalForce;
	}

	double _dt = dt / _simulationSteps;
	for (int i = 0; i < _simulationSteps; i++) {
		auto sinTheta = std::sin(_anglePos + _pi);
		auto cosTheta = std::cos(_anglePos + _pi);

		auto t0 = _massPendulum * _pendulumLength * _angleVel * _angleVel * sinTheta;
		auto t1 = _massPendulum * _g * sinTheta * cosTheta;
		auto t2 = force;
		auto n = _massCart + _massPendulum * sinTheta * sinTheta;

		auto positionAcc = (t0 + t1 + t2) / n;

		t0 = -_massPendulum * _pendulumLength * _angleVel * _angleVel * sinTheta * cosTheta;
		t1 = -(_massPendulum + _massCart) * _g * sinTheta;
		t2 = -force * cosTheta;
		n = _pendulumLength * (_massCart + _massPendulum * sinTheta * sinTheta);

		auto rotationAcc = (t0 + t1 + t2) / n;

		_angleVel *= 0.99999;
		_angleVel += rotationAcc * _dt;
		_anglePos += _angleVel * _dt;

		_cartVel *= 0.99999;
		_cartVel += positionAcc * _dt;
		_cartPos += _cartVel * _dt;

		_anglePos = fmod(_anglePos, (2 * _pi));

		if (_anglePos > _pi) {
			_anglePos -= (2 * _pi);
		} else if (_anglePos < -_pi) {
			_anglePos += (2 * _pi);
		}
	}
}





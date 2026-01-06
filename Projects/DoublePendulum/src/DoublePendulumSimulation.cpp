
#include <random>

#include "DoublePendulumSimulation.hpp"

double randomDoublePendulum(double min = 0.0, double max = 1.0) {
	static std::random_device rd;
	static std::mt19937 gen(rd());

	std::uniform_real_distribution<double> dist(min, max);
	return dist(gen);
}

DoublePendulumSimulation::DoublePendulumSimulation() {
	double offset = 0.01;

	_anglePos0 = randomDoublePendulum(_pi - offset, _pi + offset);
	_anglePos1 = randomDoublePendulum(_pi - offset, _pi + offset);

	_angleVel0 = randomDoublePendulum(-offset, offset);
	_angleVel1 = randomDoublePendulum(-offset, offset);

	_cartPos = randomDoublePendulum(-offset, offset);
	_cartVel = randomDoublePendulum(-offset, offset);

}

DoublePendulumInfo DoublePendulumSimulation::getInfo() const {
	return {_anglePos0,
	_anglePos1,
	_angleVel0,
	_angleVel1,
	_cartPos,
	_cartVel};
}

void DoublePendulumSimulation::update(double dt, double externalForce) {
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
		// All code inside this loop is made by Deepseek
		// It is also wrong..

		auto sin1 = std::sin(_anglePos0);
	    auto cos1 = std::cos(_anglePos0);
	    auto sin2 = std::sin(_anglePos1);
	    auto cos2 = std::cos(_anglePos1);

	    // Common terms
	    double m1 = _massPendulum0;
	    double m2 = _massPendulum1;
	    double l1 = _length0;
	    double l2 = _length1;
	    double M = _massCart;
	    double g = _g;

	    double omega1 = _angleVel0;
	    double omega2 = _angleVel1;

	    // Precompute some terms
	    double m2_l2_cos2 = m2 * l2 * cos2;
	    double m2_l1_cos1 = m2 * l1 * cos1;
	    double sin_diff = std::sin(_anglePos0 - _anglePos1);
	    double cos_diff = std::cos(_anglePos0 - _anglePos1);

	    // Mass matrix elements
	    double M11 = M + m1 + m2;
	    double M12 = (m1 + m2) * l1 * cos1;
	    double M13 = m2 * l2 * cos2;

	    double M21 = (m1 + m2) * l1 * cos1;
	    double M22 = (m1 + m2) * l1 * l1;
	    double M23 = m2 * l1 * l2 * cos_diff;

	    double M31 = m2 * l2 * cos2;
	    double M32 = m2 * l1 * l2 * cos_diff;
	    double M33 = m2 * l2 * l2;

	    // Coriolis and centrifugal terms
	    double C1 = -(m1 + m2) * l1 * omega1 * omega1 * sin1 - m2 * l2 * omega2 * omega2 * sin2;
	    double C2 = m2 * l1 * l2 * omega2 * omega2 * sin_diff;
	    double C3 = -m2 * l1 * l2 * omega1 * omega1 * sin_diff;

	    // Gravity terms
	    double G1 = 0;
	    double G2 = -(m1 + m2) * g * l1 * sin1;
	    double G3 = -m2 * g * l2 * sin2;

	    // External force
	    double F1 = force;
	    double F2 = 0;
	    double F3 = 0;

	    // Right-hand side vector
	    double R1 = F1 + C1 + G1;
	    double R2 = F2 + C2 + G2;
	    double R3 = F3 + C3 + G3;

	    // Solve for accelerations: M * [x_ddot, theta1_ddot, theta2_ddot]^T = R
	    // Using Cramer's rule for 3x3 system

	    double detM = M11 * (M22 * M33 - M23 * M32)
	                - M12 * (M21 * M33 - M23 * M31)
	                + M13 * (M21 * M32 - M22 * M31);

	    // Cart acceleration
	    double detA = R1 * (M22 * M33 - M23 * M32)
	                - M12 * (R2 * M33 - M23 * R3)
	                + M13 * (R2 * M32 - M22 * R3);
	    double cartAcc = detA / detM;

	    // First pendulum angular acceleration
	    double detB = M11 * (R2 * M33 - M23 * R3)
	                - R1 * (M21 * M33 - M23 * M31)
	                + M13 * (M21 * R3 - R2 * M31);
	    double theta1Acc = detB / detM;

	    // Second pendulum angular acceleration
	    double detC = M11 * (M22 * R3 - R2 * M32)
	                - M12 * (M21 * R3 - R2 * M31)
	                + R1 * (M21 * M32 - M22 * M31);
	    double theta2Acc = detC / detM;

	    // Apply damping (optional, from your single pendulum code)
	    _angleVel0 *= 0.99999;
	    _angleVel1 *= 0.99999;
	    _cartVel *= 0.99999;

	    // Update velocities and positions
	    _angleVel0 += theta1Acc * _dt;
	    _angleVel1 += theta2Acc * _dt;
	    _cartVel += cartAcc * _dt;

	    _anglePos0 += _angleVel0 * _dt;
	    _anglePos1 += _angleVel1 * _dt;
	    _cartPos += _cartVel * _dt;

	    // Normalize angles to [-π, π]
	    _anglePos0 = fmod(_anglePos0, (2 * _pi));
	    _anglePos1 = fmod(_anglePos1, (2 * _pi));

	    if (_anglePos0 > _pi) {
	        _anglePos0 -= (2 * _pi);
	    } else if (_anglePos0 < -_pi) {
	        _anglePos0 += (2 * _pi);
	    }

	    if (_anglePos1 > _pi) {
	        _anglePos1 -= (2 * _pi);
	    } else if (_anglePos1 < -_pi) {
	        _anglePos1 += (2 * _pi);
	    }
	}
}


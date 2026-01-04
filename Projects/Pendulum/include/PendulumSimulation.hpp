
#ifndef NEURALNETWORKLIB_PENDELUMSIMULATION_HPP
#define NEURALNETWORKLIB_PENDELUMSIMULATION_HPP

struct PendulumInfo {
	double anglePos;
	double angleVel;
	double cartPos;
	double cartVel;
};

class PendulumSimulation {
public:
	PendulumSimulation();
	PendulumSimulation(const PendulumInfo &info);
	PendulumSimulation(double anglePos, double angleVel, double cartPos, double cartVel);

	PendulumInfo getInfo() const;
	void update(double dt, double externalForce);

private:
	double _pendulumLength = 2.0;
	double _massCart = 1.0;
	double _massPendulum = 0.2;

	int _simulationSteps = 1;
	double _pi = 3.1415926;
	double _g = 9.81;
	double _maxForce = 10.0;

	double _anglePos;
	double _angleVel;
	double _cartPos;
	double _cartVel;
};


#endif //NEURALNETWORKLIB_PENDELUMSIMULATION_HPP

#ifndef NEURALNETWORKLIB_DOUBLEPENDULUMSIMULATION_HPP
#define NEURALNETWORKLIB_DOUBLEPENDULUMSIMULATION_HPP

struct DoublePendulumInfo {
	double anglePos0;
	double anglePos1;
	double angleVel0;
	double angleVel1;
	double cartPos;
	double cartVel;
};

class DoublePendulumSimulation {
public:
	DoublePendulumSimulation();

	DoublePendulumInfo getInfo() const;
	void update(double dt, double externalForce);

private:
	double _length0 = 2.0;
	double _length1 = 2.0;

	double _massPendulum0 = 0.5;
	double _massPendulum1 = 0.5;
	double _massCart = 1.0;

	int _simulationSteps = 1;
	double _pi = 3.1415926;
	double _g = 9.81;
	double _maxForce = 10.0;

	double _anglePos0;
	double _anglePos1;
	double _angleVel0;
	double _angleVel1;

	double _cartPos;
	double _cartVel;
};


#endif //NEURALNETWORKLIB_DOUBLEPENDULUMSIMULATION_HPP
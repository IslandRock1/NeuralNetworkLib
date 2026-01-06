
#ifndef NEURALNETWORKLIB_TESTER_HPP
#define NEURALNETWORKLIB_TESTER_HPP

#include <vector>
#include "BaseTester.hpp"
#include "PendulumSimulation.hpp"

namespace Pendulum {

	class PendulumTester: public ::Tester{
	public:

		PendulumTester();
		PendulumTester(const PendulumInfo &pendulum);
		std::vector<double> getInfo() override;
		SimulationInfo update(std::vector<double> &network, std::vector<double> &rewardModifiers) override;

		int numInputNodes() const override { return 5; }
		int numOutputNodes() const override { return 1; }

		void setValidation() override {};
		void setTraining() override {};

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
}

#endif //NEURALNETWORKLIB_TESTER_HPP
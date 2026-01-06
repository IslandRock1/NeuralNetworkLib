
#ifndef NEURALNETWORKLIB_DOUBLEPENDULUMTESTER_HPP
#define NEURALNETWORKLIB_DOUBLEPENDULUMTESTER_HPP

#include <vector>
#include "BaseTester.hpp"
#include "DoublePendulumSimulation.hpp"

namespace DoublePendulum {

	class DoublePendulumTester: public ::Tester {
	public:
		DoublePendulumTester();
		std::vector<double> getInfo() override;
		SimulationInfo update(std::vector<double>& network, std::vector<double>& rewardModifiers) override;

		int numInputNodes() const override { return 8; };
		int numOutputNodes() const override { return 1; };

		void setValidation() override {};
		void setTraining() override {};

	private:
		DoublePendulumSimulation _simulation;
		double _dt = 1.0 / 60.0;
		double _maxForce = 1.0;

		double _sumTime = 0.0;
		double _maxTime = 20.0;

		int _runningCount = 0;
		double _sumReward = 0.0;

		double rewardFunction(const std::vector<double> &constValues);
	};

}

#endif //NEURALNETWORKLIB_DOUBLEPENDULUMTESTER_HPP

#ifndef NEURALNETWORKLIB_BASETESTER_HPP
#define NEURALNETWORKLIB_BASETESTER_HPP

#include <vector>

struct SimulationInfo {
	bool isFinished;
	double progress;
	double reward;
};

class Tester {
public:
	virtual ~Tester() = default;
	virtual std::vector<double> getInfo() = 0;
	virtual SimulationInfo update(std::vector<double> &network, std::vector<double> &rewardModifiers) = 0;

	virtual int numInputNodes() const = 0;
	virtual int numOutputNodes() const = 0;
	virtual void setValidation() = 0;
	virtual void setTraining() = 0;
};

#endif //NEURALNETWORKLIB_BASETESTER_HPP

#ifndef NEURALNETWORKLIB_MNISTDIGITTESTER_HPP
#define NEURALNETWORKLIB_MNISTDIGITTESTER_HPP

#include <vector>
#include "BaseTester.hpp"

#include "mnist_loader.hpp"

namespace MNISTDigit {

	class MNISTDigitTester: public ::Tester {

	public:
		MNISTDigitTester();
		std::vector<double> getInfo() override;
		SimulationInfo update(std::vector<double>& network, std::vector<double>& rewardModifiers) override;

		int numInputNodes() const override { return 784; }; // 28x28 image size
		int numOutputNodes() const override { return 10; }; // Nums 0-9

		void setValidation() override;
		void setTraining() override;

	private:
		const mnist::MNISTDataset& _datasetTrain;
		const mnist::MNISTDataset& _datasetValidation;
		int _ix = 0;
		int numCorrect = 0;

		bool isTraining = true;
	};

}

#endif //NEURALNETWORKLIB_MNISTDIGITTESTER_HPP
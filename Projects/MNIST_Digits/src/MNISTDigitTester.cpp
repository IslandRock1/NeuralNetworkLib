
#include <iostream>

#include "MNISTDigitTester.hpp"

MNISTDigit::MNISTDigitTester::MNISTDigitTester()
	:_datasetTrain(mnist::getMNIST(
		std::string(DATA_PATH) + "/train-images.idx3-ubyte",
		std::string(DATA_PATH) + "/train-labels.idx1-ubyte"
		)), _datasetValidation(mnist::getMNISTValidation(
			std::string(DATA_PATH) + "/t10k-images.idx3-ubyte",
			std::string(DATA_PATH) + "/t10k-labels.idx1-ubyte"
		)) {
}

std::vector<double> MNISTDigit::MNISTDigitTester::getInfo() {
	if (isTraining) {
		std::vector<double> img{_datasetTrain.images[_ix].begin(), _datasetTrain.images[_ix].end()};
		return img;
	}
	std::vector<double> img{_datasetValidation.images[_ix].begin(), _datasetValidation.images[_ix].end()};
	return img;

}

SimulationInfo MNISTDigit::MNISTDigitTester::update(std::vector<double>& network, std::vector<double>& rewardModifiers) {

	int numImages;
	uint8_t correct;
	if (isTraining) {
		correct = _datasetTrain.labels[_ix];
		numImages = _datasetTrain.numImages;
	} else {
		correct = _datasetValidation.labels[_ix];
		numImages = _datasetValidation.numImages;
	}

	int ix = std::distance(network.begin(), std::max_element(network.begin(), network.end()));
	if (correct == ix) {
		numCorrect++;
	}
	_ix++;

	bool isFinished = _ix >= numImages;
	double progress = static_cast<double>(_ix) / numImages;
	double reward = static_cast<double>(numCorrect) / numImages;
	return {isFinished, progress, reward};
}

void MNISTDigit::MNISTDigitTester::setValidation() {
	isTraining = false;
}

void MNISTDigit::MNISTDigitTester::setTraining() {
	isTraining = true;
}




#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ComunicationStruct.hpp"
#include "NeuralNetwork.hpp"

#include "PendulumSimulation.hpp"
#include "PendulumTester.hpp"
#include "DoublePendulumSimulation.hpp"
#include "DoublePendulumTester.hpp"

#include "MNISTDigitTester.hpp"

#include "TrainingLib.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NeuralNetworkPython, m) {
	m.doc() = "Python bindings for Neural Network library.";

	py::class_<PythonToCPP>(m, "PythonToCPP")
		.def(py::init<>())
		.def_readwrite("projectName", &PythonToCPP::projectName)

		.def_readwrite("hiddenLayerSize", &PythonToCPP::hiddenLayerSize)
		.def_readwrite("networksPerIter", &PythonToCPP::networksPerIter)
		.def_readwrite("numSimulations", &PythonToCPP::numSimulations)
		.def_readwrite("simTime", &PythonToCPP::simTime)

		.def_readwrite("percentNetworksKept", &PythonToCPP::percentNetworksKept)
		.def_readwrite("percentNetworksNew", &PythonToCPP::percentNetworksNew)
		.def_readwrite("percentNetworksModifiable", &PythonToCPP::percentNetworksModifiable)

		.def_readwrite("percentChangeFunction", &PythonToCPP::percentChangeFunction)
		.def_readwrite("percentChangeBias", &PythonToCPP::percentChangeBias)
		.def_readwrite("percentChangeWeight", &PythonToCPP::percentChangeWeight)

		.def_readwrite("numChanges", &PythonToCPP::numChanges)
		.def_readwrite("temperature", &PythonToCPP::temperature)
		.def_readwrite("constValues", &PythonToCPP::constValues)
	;

	py::class_<CPPToPython>(m, "CPPToPython")
		.def(py::init<>())

		.def_readwrite("validationScores", &CPPToPython::validationScores)
		.def_readwrite("bestScores", &CPPToPython::bestScores)
		.def_readwrite("avgScores", &CPPToPython::avgScores)

		.def_readwrite("finishedIterations", &CPPToPython::finishedIterations)
		.def_readwrite("finishedNetworksThisIter", &CPPToPython::finishedNetworksThisIter)

		.def_readwrite("timePerMutation", &CPPToPython::timePerMutate)
		.def_readwrite("timePerNetwork", &CPPToPython::timePerNetwork)
		.def_readwrite("timePerIter", &CPPToPython::timePerIter)
		.def_readwrite("timeComputationNetwork", &CPPToPython::timeComputationNetwork)
		.def_readwrite("timeComputationSimulation", &CPPToPython::timeComputetionSimulation)
	;

	py::class_<NeuralNetwork>(m, "NeuralNetwork")
		.def(py::init<std::string>(), py::arg("Path"))
		.def("compute", &NeuralNetwork::compute, py::arg("inputValues"),
			"Compute output of network, based on input.")
	;

	py::class_<PendulumInfo>(m, "PendulumInfo")
		.def(py::init<>())

		.def_readwrite("anglePos", &PendulumInfo::anglePos)
		.def_readwrite("angleVel", &PendulumInfo::angleVel)
		.def_readwrite("cartPos", &PendulumInfo::cartPos)
		.def_readwrite("cartVel", &PendulumInfo::cartVel)
	;

	py::class_<PendulumSimulation>(m, "PendulumSimulation")
		.def(py::init<PendulumInfo>(), py::arg("PendulumInfo"))
		.def("getInfo", &PendulumSimulation::getInfo)
		.def("update", &PendulumSimulation::update)
	;

	py::class_<DoublePendulumInfo>(m, "DoublePendulumInfo")
		.def(py::init<>())

		.def_readwrite("anglePos0", &DoublePendulumInfo::anglePos0)
		.def_readwrite("anglePos1", &DoublePendulumInfo::anglePos1)

		.def_readwrite("angleVel0", &DoublePendulumInfo::angleVel0)
		.def_readwrite("angleVel1", &DoublePendulumInfo::angleVel1)

		.def_readwrite("cartPos", &DoublePendulumInfo::cartPos)
		.def_readwrite("cartVel", &DoublePendulumInfo::cartVel)
	;

	py::class_<DoublePendulumSimulation>(m, "DoublePendulumSimulation")
		.def(py::init<>())
		.def("getInfo", &DoublePendulumSimulation::getInfo)
		.def("update", &DoublePendulumSimulation::update)
	;

	py::class_<TrainingLib>(m, "TrainingLib")
		.def(py::init<PythonToCPP>(), py::arg("PythonToCPP"))
		.def("stop", &TrainingLib::stop)
		.def("setInfo", &TrainingLib::setInfo)
		.def("getInfo", &TrainingLib::getInfo)
	;

	py::class_<SimulationInfo>(m, "SimulationInfo")
		.def(py::init<>())

		.def_readwrite("isFinished", &SimulationInfo::isFinished)
		.def_readwrite("progress", &SimulationInfo::progress)
		.def_readwrite("reward", &SimulationInfo::reward)
	;

	py::class_<Pendulum::PendulumTester>(m, "PendulumTester")
		.def(py::init<>())
		.def(py::init<PendulumInfo>(), py::arg("PendulumInfo"))

		.def("getInfo", &Pendulum::PendulumTester::getInfo)
		.def("update", &Pendulum::PendulumTester::update)
	;

	py::class_<DoublePendulum::DoublePendulumTester>(m, "DoublePendulumTester")
		.def(py::init<>())

		.def("getInfo", &DoublePendulum::DoublePendulumTester::getInfo)
		.def("update", &DoublePendulum::DoublePendulumTester::update)
	;

	py::class_<MNISTDigit::MNISTDigitTester>(m, "MNISTDigitTester")
		.def(py::init<>())

		.def("getInfo", &MNISTDigit::MNISTDigitTester::getInfo)
		.def("update", &MNISTDigit::MNISTDigitTester::update)
	;

	py::class_<mnist::MNISTDataset>(m, "MNISTDataset")
		.def_readwrite("numImages", &mnist::MNISTDataset::numImages)
		.def_readwrite("rows", &mnist::MNISTDataset::rows)
		.def_readwrite("cols", &mnist::MNISTDataset::cols)
		.def_readwrite("images", &mnist::MNISTDataset::images)
		.def_readwrite("labels", &mnist::MNISTDataset::labels)
	;

	m.def("loadMNISTValidation", &mnist::getMNISTValidation, py::arg("imageFile"), py::arg("labelFile"));

}
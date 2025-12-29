
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ComunicationStruct.hpp"
#include "NeuralNetwork.hpp"
#include "PendulumSimulation.hpp"

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
}
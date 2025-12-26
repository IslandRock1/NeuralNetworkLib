
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ComunicationStruct.hpp"
#include "NeuralNetwork.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NeuralNetworkPython, m) {
	m.doc() = "Python bindings for Neural Network library.";

	py::class_<PythonToCPPInit>(m, "PythonToCPPInit")
		.def(py::init<>())
		.def_readwrite("projectName", &PythonToCPPInit::projectName)
		.def_readwrite("hiddenLayerSize", &PythonToCPPInit::hiddenLayerSize)
		.def_readwrite("networksPerIter", &PythonToCPPInit::networksPerIter)
		.def_readwrite("percentNetworksKept", &PythonToCPPInit::percentNetworksKept)
		.def_readwrite("percentNetworksNew", &PythonToCPPInit::percentNetworksNew)
		.def_readwrite("percentNetworksModifiable", &PythonToCPPInit::percentNetworksModifiable)
		.def_readwrite("percentChangeFunction", &PythonToCPPInit::percentChangeFunction)
		.def_readwrite("percentChangeBias", &PythonToCPPInit::percentChangeBias)
		.def_readwrite("percentChangeWeight", &PythonToCPPInit::percentChangeWeight)
	;

	py::class_<PythonToCPPIter>(m, "PythonToCPPIter")
		.def(py::init<>())
		.def_readwrite("numChanges", &PythonToCPPIter::numChanges)
		.def_readwrite("temperature", &PythonToCPPIter::temperature)
		.def_readwrite("constValues", &PythonToCPPIter::constValues)
	;

	py::class_<CPPToPython>(m, "CPPToPython")
		.def(py::init<>())
		.def_readwrite("bestScores", &CPPToPython::bestScores)
		.def_readwrite("avgScores", &CPPToPython::avgScores)
		.def_readwrite("finishedNetworksThisIter", &CPPToPython::finishedNetworksThisIter)
		.def_readwrite("timePerNetwork", &CPPToPython::timePerNetwork)
		.def_readwrite("timePerIter", &CPPToPython::timePerIter)
		.def_readwrite("timeComputationNetwork", &CPPToPython::timeComputationNetwork)
		.def_readwrite("timeComputationSimulation", &CPPToPython::timeComputetionSimulation)
	;

	py::class_<NeuralNetwork>(m, "NeuralNetwork")
		.def(py::init<std::vector<int>>(), py::arg("Layer Sizes"))
		.def("compute", &NeuralNetwork::compute, py::arg("inputValues"),
			"Compute output of network, based on input.")
	;
}
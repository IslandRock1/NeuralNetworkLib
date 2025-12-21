
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ModificationOptions.hpp"
#include "NeuralNetwork.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NeuralNetworkPython, m) {
	m.doc() = "Python bindings for Neural Network library.";

	py::class_<ModificationOptions>(m, "ModificationOptions")
		.def(py::init())
	;

	py::class_<NeuralNetwork>(m, "NeuralNetwork")
		.def(py::init<std::vector<int>>(), py::arg("Layer Sizes"))
		.def("compute", &NeuralNetwork::compute, py::arg("inputValues"),
			"Compute output of network, based on input.")
		.def("executeRandomChange", &NeuralNetwork::executeRandomChange,
			py::arg("modificationOptions"), "Do random modifications to the network.")
	;
}
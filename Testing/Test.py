
from time import perf_counter
from math import sin, cos

from NeuralNetworkPython import CPPToPython, PythonToCPP, NeuralNetwork, PendulumInfo, PendulumSimulation

pendulumInfo = PendulumInfo()
pendulumInfo.anglePos = 3.1415926
pendulumInfo.angleVel = 0.0
pendulumInfo.cartPos = 0.0
pendulumInfo.cartVel = 0.0

pendulumSim = PendulumSimulation(pendulumInfo)

print()

def convertInfo(pendulumInfo: PendulumInfo):
    pendulumX = sin(pendulumInfo.anglePos)
    pendulumY = cos(pendulumInfo.anglePos)
    return [pendulumX, pendulumY, pendulumInfo.angleVel, pendulumInfo.cartPos, pendulumInfo.cartVel]

for _ in range(10):
    nn = NeuralNetwork("Testing/1")
    inputValues = convertInfo(pendulumSim.getInfo())

    output = nn.compute(inputValues)
    pendulumSim.update(1.0 / 60.0, 100)
    print(inputValues)
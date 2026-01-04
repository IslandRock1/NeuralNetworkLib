
from time import perf_counter, sleep
from math import sin, cos

from NeuralNetworkPython import CPPToPython, PythonToCPP, NeuralNetwork, PendulumInfo, PendulumSimulation, TrainingLib

print()

def testTraining():
    def printInfo(info: CPPToPython):
        print(f"Best scores: {info.bestScores}")


    pythonToCPP = PythonToCPP()
    pythonToCPP.projectName = "Pendulum"

    pythonToCPP.hiddenLayerSize = [20, 50, 20]
    pythonToCPP.networksPerIter = 1000
    pythonToCPP.numSimulations = 1
    pythonToCPP.simTime = 20.0

    pythonToCPP.percentNetworksKept = 0.1
    pythonToCPP.percentNetworksNew = 0.2
    pythonToCPP.percentNetworksModifiable = 0.3

    pythonToCPP.percentChangeFunction = 0.01
    pythonToCPP.percentChangeBias = 0.09
    pythonToCPP.percentChangeWeight = 0.9

    pythonToCPP.numChanges = 1
    pythonToCPP.temperature = 1.0
    pythonToCPP.constValues = [1.0, 0.0, 0.0, 0.0]

    trainingLib = TrainingLib(pythonToCPP)

    t0 = perf_counter()
    while (perf_counter() - t0 < 100):
        info = trainingLib.getInfo()
        printInfo(info)
        sleep(1.0)

    trainingLib.stop()

def testUsing():
    pendulumInfo = PendulumInfo()
    pendulumInfo.anglePos = 3.1415926
    pendulumInfo.angleVel = 0.0
    pendulumInfo.cartPos = 0.0
    pendulumInfo.cartVel = 0.0

    pendulumSim = PendulumSimulation(pendulumInfo)

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

def main():
    testTraining()

if (__name__ == "__main__"): main()
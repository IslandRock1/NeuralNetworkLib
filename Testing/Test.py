
from time import perf_counter

from NeuralNetworkPython import NeuralNetwork


print()

for _ in range(10):
    nn = NeuralNetwork([5, 100, 1000, 10000, 1000, 100, 1])
    inputValues = [0.0, 0.0, 0.0, 0.0, 0.0]

    t0 = perf_counter()
    output = nn.compute(inputValues)
    t1 = perf_counter()
    print(f"Compute time: {t1 - t0} | {output}")
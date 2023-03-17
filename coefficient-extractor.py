import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from decimal import Decimal, getcontext
import math


if __name__ == "__main__":
    data = "/Users/alexradcliffe/kcl/hyperbolic-python/macData/"
    data = "/Users/alexradcliffe/kcl/hyperbolic-python/clusterData240/"
    data = sys.argv[1]
    outputJsons = f"{data}/outputJsons"
    files = os.listdir(outputJsons)
    bounds = {}
    coefficients = {}
    for file in files:
        with open(f"{outputJsons}/{file}") as f:
            j = json.load(f)
        lMax = j["lmax"]
        bounds[lMax] = j["bound"]
        coefficients[lMax] = {}
        for i, coeff in enumerate(j["coefficients"]):
            coefficients[lMax][i] = coeff
    lMaxs = sorted(bounds.keys())
    bounds = [bounds[i] for i in lMaxs]
    coefficients = [coefficients[i] for i in lMaxs]
    with open(f"{data}/lMaxs.json", "w") as f:
        json.dump(lMaxs, f, indent = 4)
    with open(f"{data}/bounds.json", "w") as f:
        json.dump(bounds, f, indent = 4)
    with open(f"{data}/coefficients.json", "w") as f:
        json.dump(coefficients, f, indent = 4)

    coefficients = coefficients[:-3]
    
    getcontext().prec = 500
    sparseMatrix = []
    lMaxs = []
    for coeffs in coefficients:
        lMax = len(coeffs) - 1
        lMaxs.append(lMax)
        for index, coeff in coeffs.items():
            sparseMatrix.append(((lMax, index), Decimal(coeff)))
    lMaxMax = max(lMaxs)
    print(sparseMatrix)
    coeffsByLMax = {lMax : {} for lMax in lMaxs}
    coeffsByIndex = {index : {} for index in range(lMaxMax + 1)}
    for (lMax, index), coeff in sparseMatrix:
        coeffsByLMax[lMax][index] = coeff
        coeffsByIndex[index][lMax] = coeff
    fig, axs = plt.subplots(len(lMaxs), figsize=(10,100))
    for i, lMax in enumerate(sorted(lMaxs)):
        coeffs = coeffsByLMax[lMax]
        xs, ys = [], []
        for x, y in coeffs.items():
            xs.append(x)
            ys.append(Decimal(y))
        ys = [math.log(abs(float(y))) for y in ys]
        axs[i].plot(xs, ys)
        axs[i].set_ylabel("lMax = " + str(lMax))
    fig.tight_layout(pad=5.0)
    plt.savefig("byLMax.png")


    fig, axs = plt.subplots(lMaxMax + 1, figsize=(10,100))
    for i in range(lMaxMax + 1):
        coeffs = coeffsByIndex[i]
        xs, ys = [], []
        for x, y in coeffs.items():
            xs.append(x)
            ys.append(Decimal(y))
        ys = [math.log(abs(float(y))) for y in ys]
        axs[i].plot(xs, ys)
        axs[i].set_ylabel("$\\alpha_{" + str(i) + "}$")
    fig.tight_layout(pad=5.0)
    plt.savefig("byIndex.png")

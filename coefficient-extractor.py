import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from decimal import Decimal, getcontext
import math

def processJsons(data):
    outputJsons = f"{data}/outputJsons"
    files = os.listdir(outputJsons)
    bounds = {}
    possibleKeys = ["coefficients", "qs", "rs", "alphas", "betas", "gammas", "deltas"]
    keyDicts = {key : {} for key in possibleKeys}
    for file in files:
        with open(f"{outputJsons}/{file}") as f:
            j = json.load(f)
        lMax = j["lmax"]
        bounds[lMax] = j["bound"]
        lMaxs = sorted(bounds.keys())
        for key in possibleKeys:
            if key in j:
                keyDicts[key][lMax] = {}
                for i, coeff in enumerate(j[key]):
                    keyDicts[key][lMax][i] = coeff
    with open(f"{data}/lMaxs.json", "w") as f:
        json.dump(lMaxs, f, indent = 4)
    boundList = [bounds[i] for i in lMaxs]
    with open(f"{data}/bounds.json", "w") as f:
        json.dump(boundList, f, indent = 4)
    keyLists = {key : [] for key in possibleKeys}
    for key in possibleKeys:
        if len(keyDicts[key]) != 0:
            keyLists[key] = [keyDicts[key][i] for i in lMaxs]
            with open(f"{data}/{key}.json", "w") as f:
                json.dump(keyLists[key], f, indent = 4)
    return lMaxs, keyLists

def coeffsToSparseMatrices(lMaxs, coefficients):
    sparseMatrix = []
    for lMax, coeffs in zip(lMaxs, coefficients):
        for index, coeff in coeffs.items():
            sparseMatrix.append(((lMax, index), Decimal(coeff)))
    return sparseMatrix

def coeffMatrices(lMaxs, coefficients):
    sparseMatrix = coeffsToSparseMatrices(lMaxs, coefficients)
    lMaxMax = max(lMaxs)
    coeffsByLMax = {lMax : {} for lMax in lMaxs}
    coeffsByIndex = {index : {} for index in range(lMaxMax + 1)}
    for (lMax, index), coeff in sparseMatrix:
        coeffsByLMax[lMax][index] = coeff
        coeffsByIndex[index][lMax] = coeff
    return coeffsByLMax, coeffsByIndex

def makePlots(indices, dataByIndices, indexName, dataName, folder):
    fig, axs = plt.subplots(len(indices), figsize=(10,100))
    for i, index in enumerate(sorted(indices)):
        coeffs = dataByIndices[index]
        xs, ys = [], []
        for x, y in coeffs.items():
            xs.append(x)
            ys.append(Decimal(y))
        ys = [math.log(abs(float(y))) for y in ys]
        axs[i].plot(xs, ys)
        axs[i].set_ylabel(f"{indexName} = " + str(index))
    fig.tight_layout(pad=5.0)
    plt.savefig(f"{folder}/{dataName}by{indexName}.png")


if __name__ == "__main__":
    data = "/Users/alexradcliffe/kcl/hyperbolic-python/macData/"
    data = "/Users/alexradcliffe/kcl/hyperbolic-python/clusterData240/"
    data = sys.argv[1]
    if data[-1] == "/":
        data = data[::-1]
    lMaxs, keyLists = processJsons(data)
    print(keyLists["alphas"])
    
    getcontext().prec = 500
    possibleKeys = ["coefficients", "qs", "rs", "alphas", "betas", "gammas", "deltas"]
    for key in possibleKeys:
        if keyLists[key]:
            coeffsByLMax, coeffsByIndex = coeffMatrices(lMaxs, keyLists[key])
            makePlots(lMaxs, coeffsByLMax, "lMax", key, data)
            lMaxMax = max(lMaxs)
            indices = list(range(lMaxMax + 1))
            makePlots(indices, coeffsByIndex, "index", key, data)

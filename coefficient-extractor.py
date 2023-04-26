import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from decimal import Decimal, getcontext
import math

def processJsons(data):
    outputJsons = f"{data}/outputJsons"
    print(outputJsons)
    files = os.listdir(outputJsons)
    bounds = {}
    coefficients = {}
    qs = {}
    rs = {}
    for file in files:
        with open(f"{outputJsons}/{file}") as f:
            j = json.load(f)
        lMax = j["lmax"]
        bounds[lMax] = j["bound"]
        if "coefficients" in j:
            coefficients[lMax] = {}
            for i, coeff in enumerate(j["coefficients"]):
                coefficients[lMax][i] = coeff
            lMaxs = sorted(bounds.keys())
            boundList = [bounds[i] for i in lMaxs]
            coefficientList = [coefficients[i] for i in lMaxs]
            qList = []
            rList = []
            with open(f"{data}/lMaxs.json", "w") as f:
                json.dump(lMaxs, f, indent = 4)
            with open(f"{data}/bounds.json", "w") as f:
                json.dump(boundList, f, indent = 4)
            with open(f"{data}/coefficients.json", "w") as f:
                json.dump(coefficientList, f, indent = 4)
        else:
            qs[lMax] = {}
            rs[lMax] = {}
            for i, q in enumerate(j["qs"]):
                qs[lMax][i] = q
            for i, r in enumerate(j["rs"]):
                rs[lMax][i] = r
            lMaxs = sorted(bounds.keys())
            boundList = [bounds[i] for i in lMaxs]
            coefficientList = []
            qList = [qs[i] for i in lMaxs]
            rList = [rs[i] for i in lMaxs]
            with open(f"{data}/lMaxs.json", "w") as f:
                json.dump(lMaxs, f, indent = 4)
            with open(f"{data}/bounds.json", "w") as f:
                json.dump(boundList, f, indent = 4)
            with open(f"{data}/qs.json", "w") as f:
                json.dump(qList, f, indent = 4)
            with open(f"{data}/rs.json", "w") as f:
                json.dump(rList, f, indent = 4)
    return lMaxs, coefficientList, qList, rList

def coeffsToSparseMatrices(coefficients):
    sparseMatrix = []
    lMaxs = []
    for coeffs in coefficients:
        print("coeffs", coeffs)
        lMax = len(coeffs) - 1
        lMaxs.append(lMax)
        for index, coeff in coeffs.items():
            sparseMatrix.append(((lMax, index), Decimal(coeff)))
    return lMaxs, sparseMatrix

def coeffMatrices(coefficients):
    lMaxs, sparseMatrix = coeffsToSparseMatrices(coefficients)
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
    lMaxs, coefficients, qs, rs = processJsons(data)
    
    getcontext().prec = 500
    if coefficients:
        coeffsByLMax, coeffsByIndex = coeffMatrices(coefficients)
        makePlots(lMaxs, coeffsByLMax, "lMax", "coeffs", data)
        lMaxMax = max(lMaxs)
        indices = list(range(lMaxMax + 1))
        makePlots(indices, coeffsByIndex, "index", "coeffs", data)
    else:
        qsByLMax, qsByIndex = coeffMatrices(qs)
        makePlots(lMaxs, qsByLMax, "lMax", "qs", data)
        lMaxMax = max(lMaxs)
        indices = list(range(lMaxMax + 1))
        makePlots(indices, qsByIndex, "index", "qs", data)
        rsByLMax, rsByIndex = coeffMatrices(rs)
        makePlots(lMaxs, rsByLMax, "lMax", "rs", data)
        makePlots(indices, rsByIndex, "index", "rs", data)

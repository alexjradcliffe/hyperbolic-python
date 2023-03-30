import json
import os
import numpy as np
from decimal import Decimal, getcontext
import operator
import sys
from hyperbolic import F, Polynomial, jsonInput

def read_output(lmax, directory):
    with open(f"{directory}/tmp/gapPy{lmax}_out/out.txt") as f:
        result = f.readlines()[0]
    if result == 'terminateReason = "found dual feasible solution";\n':
        return True
    elif result == 'terminateReason = "found primal feasible solution";\n':
        return False
    else:
        raise Exception("SDPB returned a weird response.")

def max_index(normalization):
    return np.argmax(np.abs(list(map(Decimal, normalization))))

def outputFromTxt(lmax, normalization, directory):
    with open(f"{directory}/tmp/gapPy{lmax}_out/y.txt") as f:
        y = list(map(Decimal, f.read().split()[2:]))
    n = list(map(Decimal, normalization))
    index = max_index(n)
    n0 = n.pop(index)
    y.insert(index, (1 - sum(map(operator.mul, n, y)) / n0))
    return y

def gap_test(lambdaGap, lmax, sdpbPrec, procsPerNode, mac):
    if mac:
        directory = "/Users/alexradcliffe/kcl/hyperbolic-python"
    else:
        directory = "/users/k21187236/scratch/hyperbolic-python"
    os.chdir(directory)
    nn = 6
    print("Testing", lambdaGap)
    normalization = [str(F(nn, 2 * nn + l)(0)) for l in range(lmax + 1)]
    objective = ["0"] * (lmax + 1)
    polynomials = [[["0"] if i != j else ["-1"] for i in range(lmax + 1)] for j in list(range(0, lmax, 2))] + [
                #    [[str(F(nn, 2 * nn + l)(0))] for l in range(lmax + 1)]] + [
                   [list(map(str, F(nn, 2 * nn + l).shift(-lambdaGap).coefficients)) for l in range(lmax + 1)]]
    with open(f"{directory}/tmp/gapPy{lmax}.json", "w") as f:
        json.dump(jsonInput(objective, normalization, polynomials), f, indent=4)
    if mac:
        sdp2input = f"/usr/local/bin/docker run -v {directory}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdp2input --precision={sdpbPrec} --input=/usr/local/share/sdpb/gapPy{lmax}.json --output=/usr/local/share/sdpb/gapPy{lmax}"
        sdpb = f"/usr/local/bin/docker run -v {directory}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdpb  --findPrimalFeasible --findDualFeasible --precision={sdpbPrec} --procsPerNode={procsPerNode} -s /usr/local/share/sdpb/gapPy{lmax}"
    else:
        sdp2input = f"sdp2input --precision={sdpbPrec} --input={directory}/tmp/gapPy{lmax}.json --output={directory}/tmp/gapPy{lmax}"
        sdpb = f"sdpb  --findPrimalFeasible --findDualFeasible --precision={sdpbPrec} --procsPerNode={procsPerNode} -s {directory}/tmp/gapPy{lmax}"
    os.system(sdp2input)
    os.system(sdpb)
    return read_output(lmax, directory)

# for i in range(2, 31):
#     print(f"lmax={i}")
#     OPE_bound(i, 200)

def binarySearch(lambdaMin, lambdaMax, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, mac):
    if lambdaMax - lambdaMin < gapPrec:
        if mac:
            directory = "/Users/alexradcliffe/kcl/hyperbolic-python"
        else:
            directory = "/users/k21187236/scratch/hyperbolic-python"
        bound = lambdaMax
        gap_test(bound, lmax, sdpbPrec, procsPerNode, mac)
        nn = 6
        normalization = [str(F(nn, 2 * nn + l)(0)) for l in range(lmax + 1)]
        coefficients = outputFromTxt(lmax, normalization, directory)
        jsonResults = {"lmax" : lmax,
                       "gapPrec" : float(gapPrec),
                       "pythonPrec" : pythonPrec,
                       "sdpbPrec" : sdpbPrec,
                       "procsPerNode" : procsPerNode,
                       "bound" : str(bound),
                       "coefficients" : list(map(str, coefficients))}
        with open(f"{directory}/outputJsons/gapPy{lmax}_out.json", "w") as f:
            json.dump(jsonResults, f, indent=4)
        return lambdaMax
    else:
        lambdaMiddle = (lambdaMin + lambdaMax) / 2
        if gap_test(lambdaMiddle, lmax, sdpbPrec, procsPerNode, mac):
            lambdaMax = lambdaMiddle
        else:
            lambdaMin = lambdaMiddle
        return binarySearch(lambdaMin, lambdaMax, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, mac)

if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    pythonPrec = config["pythonPrec"]
    getcontext().prec = pythonPrec
    lambdaMin = Decimal(config["lambdaMin"])
    lambdaMax = Decimal(config["lambdaMax"])
    gapPrec = Decimal(config["gapPrec"])
    lmaxLower = config["lmaxLower"]
    lmaxUpper = config["lmaxUpper"]
    sdpbPrec = config["sdpbPrec"]
    procsPerNode = config["procsPerNode"]
    oddOnly = config["oddOnly"]
    mac = config["mac"]
    step = 2 if oddOnly else 1
    if oddOnly and lmaxLower % 2 == 0:
        lmaxLower += 1
    for lmax in range(lmaxLower, lmaxUpper + 1, step):
        print(lambdaMin, lambdaMax, gapPrec)
        print(binarySearch(lambdaMin, lambdaMax, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, mac))
import json
import os
import numpy as np
from decimal import Decimal, getcontext
import operator
import sys
from hyperbolic import F, Polynomial, jsonInput

def read_output(lmax, workingDir):
    with open(f"{workingDir}/tmp/gapPy{lmax}_out/out.txt") as f:
        result = f.readlines()[0]
    if result == 'terminateReason = "found dual feasible solution";\n':
        return True
    elif result == 'terminateReason = "found primal feasible solution";\n':
        return False
    else:
        raise Exception("SDPB returned a weird response.")

def max_index(normalization):
    return np.argmax(np.abs(list(map(Decimal, normalization))))

def outputFromTxt(lmax, normalization, workingDir):
    with open(f"{workingDir}/tmp/gapPy{lmax}_out/y.txt") as f:
        y = list(map(Decimal, f.read().split()[2:]))
    norm = list(map(Decimal, normalization))
    index = max_index(norm)
    norm0 = norm.pop(index)
    y.insert(index, (1 - sum(map(operator.mul, norm, y)) / norm0))
    return y

def gap_test(lambdaGap, n, ln, lmax, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac):
    os.chdir(workingDir)
    print("Testing", lambdaGap)
    if ln == 1:
        normalization = [str(F(n, 2 * n + l)(0)) for l in range(lmax + 1)]
        objective = ["0"] * (lmax + 1)
        polynomials = [[["0"] if i != j else ["-1"] for i in range(lmax + 1)] for j in list(range(0, lmax, 2))] + [
                    #    [[str(F(n, 2 * n + l)(0))] for l in range(lmax + 1)]] + [
                       [list(map(str, F(n, 2 * n + l).shift(-lambdaGap).coefficients)) for l in range(lmax + 1)]]
    else:
        normalization = [str(Decimal(-(1 + ln) / ln) * F(n, 2 * n + l)(0)) for l in range(lmax + 1)] + [str(-F(n, 2 * n + l)(0)) for l in range(lmax + 1)]
        objective = ["0"] * (2 * lmax + 2)
        polynomials = ([[["0"] if i != j else ["1"] for i in range(lmax + 1)] + ([["0"]] * (lmax + 1)) for j in list(range(0, lmax + 1, 2))] +
                       [([["0"]] * (lmax + 1)) + [["0"] if i != j else ["1"] for i in range(lmax + 1)] for j in list(range(1, lmax + 1, 2))] +
                       [[list(map(str, (Polynomial([Decimal((1 - ln) / ln)]) * F(n, 2 * n + l).shift(-lambdaGap)).coefficients)) for l in range(lmax + 1)] +
                        [list(map(str, (F(n, 2 * n + l).shift(-lambdaGap).coefficients))) for l in range(lmax + 1)]] +
                       [[list(map(str, (Polynomial([Decimal(-(1 + ln) / ln)]) * F(n, 2 * n + l).shift(-lambdaGap)).coefficients)) for l in range(lmax + 1)] +
                        [list(map(str, ((Polynomial([Decimal(-(1 + ln) / ln)]) * F(n, 2 * n + l).shift(-lambdaGap)).coefficients))) for l in range(lmax + 1)]])
        
    with open(f"{workingDir}/tmp/gapPy{lmax}.json", "w") as f:
        json.dump(jsonInput(objective, normalization, polynomials), f, indent=4)
    if mac:
        sdp2input = f"/usr/local/bin/docker run -v {workingDir}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdp2input --precision={sdpbPrec} --input=/usr/local/share/sdpb/gapPy{lmax}.json --output=/usr/local/share/sdpb/gapPy{lmax}"
        sdpb = f"/usr/local/bin/docker run -v {workingDir}/tmp/:/usr/local/share/sdpb wlandry/sdpb:2.5.1 mpirun --allow-run-as-root -n 4 sdpb  --findPrimalFeasible --findDualFeasible --precision={sdpbPrec} --procsPerNode={procsPerNode} --dualityGapThreshold={dualityGapThreshold} --primalErrorThreshold={dualityGapThreshold} --dualErrorThreshold={dualityGapThreshold} -s /usr/local/share/sdpb/gapPy{lmax}"
    else:
        sdp2input = f"sdp2input --precision={sdpbPrec} --input={workingDir}/tmp/gapPy{lmax}.json --output={workingDir}/tmp/gapPy{lmax}"
        sdpb = f"sdpb  --findPrimalFeasible --findDualFeasible --precision={sdpbPrec} --procsPerNode={procsPerNode} --dualityGapThreshold={dualityGapThreshold} --primalErrorThreshold={dualityGapThreshold} --dualErrorThreshold={dualityGapThreshold} -s {workingDir}/tmp/gapPy{lmax}"
    os.system(sdp2input)
    os.system(sdpb)
    return read_output(lmax, workingDir)

# for i in range(2, 31):
#     print(f"lmax={i}")
#     OPE_bound(i, 200)

def binarySearch(lambdaMin, lambdaMax, n, ln, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac):
    if lambdaMax - lambdaMin < gapPrec:
        bound = lambdaMax
        gap_test(bound, n, ln, lmax, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac)
        if ln == 1:
            normalization = [str(F(nn, 2 * nn + l)(0)) for l in range(lmax + 1)]
        else:
            normalization = [str(Decimal(-(1 + ln) / ln) * F(n, 2 * n + l)(0)) for l in range(lmax + 1)] + [str(-F(n, 2 * n + l)(0)) for l in range(lmax + 1)]
        coefficients = outputFromTxt(lmax, normalization, workingDir)
        jsonResults = {"n" : n,
                       "ln" : ln,
                       "lmax" : lmax,
                       "gapPrec" : float(gapPrec),
                       "pythonPrec" : pythonPrec,
                       "sdpbPrec" : sdpbPrec,
                       "procsPerNode" : procsPerNode,
                       "bound" : str(bound),
                       "coefficients" : list(map(str, coefficients))}
        with open(f"{workingDir}/outputJsons/gapPy{lmax}_out.json", "w") as f:
            json.dump(jsonResults, f, indent=4)
        return lambdaMax
    else:
        lambdaMiddle = (lambdaMin + lambdaMax) / 2
        if gap_test(lambdaMiddle, n, ln, lmax, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac):
            lambdaMax = lambdaMiddle
        else:
            lambdaMin = lambdaMiddle
        return binarySearch(lambdaMin, lambdaMax, n, ln, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac)

if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    pythonPrec = config["pythonPrec"]
    getcontext().prec = pythonPrec
    lambdaMin = Decimal(config["lambdaMin"])
    lambdaMax = Decimal(config["lambdaMax"])
    n = config["n"]
    ln = config["ln"]
    gapPrec = Decimal(config["gapPrec"])
    lmaxLower = config["lmaxLower"]
    lmaxUpper = config["lmaxUpper"]
    sdpbPrec = config["sdpbPrec"]
    procsPerNode = config["procsPerNode"]
    dualityGapThreshold = config["dualityGapThreshold"]
    oddOnly = config["oddOnly"]
    workingDir = config["workingDir"]
    mac = config["mac"]
    step = 2 if oddOnly else 1
    if oddOnly and lmaxLower % 2 == 0:
        lmaxLower += 1
    for lmax in range(lmaxLower, lmaxUpper + 1, step):
        print(lambdaMin, lambdaMax, gapPrec)
        print(binarySearch(lambdaMin, lambdaMax, n, ln, gapPrec, lmax, pythonPrec, sdpbPrec, procsPerNode, dualityGapThreshold, workingDir, mac))